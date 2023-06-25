# See https://www.tensorflow.org/tutorials/audio/music_generation
# Temperature https://www.tensorflow.org/text/tutorials/text_generation
import pandas as pd
import datetime
import glob
import numpy as np
import pathlib
import pretty_midi
import tensorflow as tf
import os

from analyze import midi_to_notes, plot_notes, plot_distributions, plot_loss

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
pretty_midi.pretty_midi.MAX_TICK = 1e10
LOSS_WEIGHTS = {
    'pitch': 0.053,  # 0.05
    'step': 1.0,  # 1.0
    'duration': 1.0,  # 1.0
}
KEY_ORDER = ['pitch', 'step', 'duration']
FILE_NAME = "{identifier}/{time}/{name}.{extension}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
POSITIVE_PRESSURE_STRENGTH = 15
VOCAB_SIZE = 128
MAESTRO = pathlib.Path('data/maestro-v2.0.0')


def check_dataset_existence():
    """
    Checks if datasets exist. If not, dataset is downloaded.
    :return:
    """
    if not MAESTRO.exists():
        print(f"Downloading {MAESTRO}...")
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )


def get_ym2413_file_names_by_emotion(emotion, toptag=True):  # ONLY FOR YM2413 DATABASE
    """
    Gets filenames based on emotion from ym2413. However, MIDI files from that dataset can include larger gaps and
    are in general atypical.
    :param emotion: Emotion tag to filter for [cheerful, tense, peaceful, creepy, depressed, comic, serious, touching,
    speedy, dreamy, cute, grand, exciting, rhythmic, boring, bizarre, frustrating, cold, calm]
    :param toptag: If true, only check top tag (tag people voted most for).
    :return:
    """
    df = pd.read_csv('data/YM2413-MDB-v1.0.2/emotion_annotation/verified_annotation_old.csv', delimiter=',')

    # Filter DataFrame based on emotion
    if toptag:
        filtered_data = df[df['toptag_eng_verified'] == emotion]
    else:
        filtered_data = df[df['verified_tags'].str.contains(emotion)]

    # wav to midi
    file_names = [f"data/YM2413-MDB-v1.0.2/midi/adjust_tempo_remove_delayed_inst/{name[:-3]}mid" for name in
                  filtered_data['fname']]
    return file_names


def notes_write_to_midi(
        notes: pd.DataFrame,
        out_file: str,
        instrument_name: str,
        velocity: int = 100,  # note loudness; currently ignored during training. See
) -> pretty_midi.PrettyMIDI:
    """
    Converts notes to midi file out_file.
    :param notes: notes to write
    :param out_file: file to write to
    :param instrument_name: instrument being used
    :param velocity: velocity of every note (all notes have the same velocity as velocity is currently not trained)
    :return:
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)

    sub_dir_name = os.path.dirname(out_file)
    if not os.path.isdir(sub_dir_name):
        dir_name = os.path.dirname(sub_dir_name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        os.mkdir(sub_dir_name)
    pm.write(out_file)


def create_sequences(
        dataset: tf.data.Dataset,
        seq_length: int,
) -> tf.data.Dataset:
    """
    Creates a number of 1 shifted sequences where the "last element" becomes the label. For example:
    0 1 2 [3] | 4 5
    1 2 3 [4] | 5
    2 3 4 [5]
    That way we can predict a value based on the sequence and see if we hit the last element later.
    For more information see: See https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window
    :param dataset: note data
    :param seq_length: length of the sequence
    :return:
    """
    seq_length = seq_length + 1  # give space for labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)  # first element of a line is k*shift; following elem is +stride
    sequences = windows.flat_map(lambda x: x.batch(seq_length, drop_remainder=True))  # batch windows

    # Split the labels
    def split_labels(sequences):
        """
        Split sequences into pitch-normalized note sequence and labels
        :param sequences: sequence to split
        :return: tuple (sequence, label)
        """
        inputs = sequences[:-1]  # take everything but last element of a sequence as input
        labels_dense = sequences[-1]  # take last element of a sequence as label
        labels = {key: labels_dense[i] for i, key in enumerate(KEY_ORDER)}
        return inputs / [VOCAB_SIZE, 1.0, 1.0], labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Just like mse, but the absolute of negative numbers times constant POSITIVE_PRESSURE_STRENGTH
    is added before calculating the mean. This discourages learning negative values.
    :param y_true:
    :param y_pred:
    :return:
    """
    mse = (y_true - y_pred) ** 2
    positive_pressure = POSITIVE_PRESSURE_STRENGTH * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)  # reduce to single value by using mean


def predict_next_note(
        notes: np.ndarray,
        keras_model: tf.keras.Model,
        temperature: float = 1.0) -> (int, float, float):
    """
    Predicts a single note given a sequence of notes.
    :param notes: sequence of notes
    :param keras_model: model to create notes from
    :param temperature: randomness factor for pitch selection
    :return: a note (pitch, step, duration)
    """

    if temperature <= 0:
        raise ValueError("Temperature must be >0!")

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be >0
    if step < 0:
        print(f"Step <0 during prediction!")
    step = tf.maximum(0, step)
    if duration < 0:
        print(f"Duration <0 during prediction!")
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def generate_notes_from_file(model, base_file, seq_length=25):
    """

    :param model:
    :param base_file:
    :param seq_length:
    :return:
    """
    print("Base File: " + base_file)
    raw_notes = midi_to_notes(base_file)
    pm = pretty_midi.PrettyMIDI(base_file)
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    sample_notes = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)
    return instrument_name, generate_notes(model=model,
                                           input_notes=(sample_notes[:seq_length] / np.array([VOCAB_SIZE, 1, 1])))


def generate_notes(model, input_notes, temperature=2.0, num_predictions=240):
    """
    Given a trained model, sequence of input_notes and an instrument name, a sequence of notes is generated.
    :param model: trained model
    :param input_notes: sequence of notes [(pitch, step, duration), ...]
    :param temperature: randomness factor for pitch selection
    :param num_predictions: number of notes to predict for returned generated note sequence.
    :return: sequence of generated notes
    """
    for x in range(1):  # currently one file is written
        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = predict_next_note(input_notes, model, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*KEY_ORDER, 'start', 'end'))
    return generated_notes


def train(identifier, filenames, seq_length=25, learning_rate=0.005, epochs=25, loss_weights=LOSS_WEIGHTS,
          num_files=15, model=None):
    """
    Trains and returns a model on files filenames
    :param identifier: identifier for saved files
    :param filenames: names of files to use for training
    :param seq_length: length of sequences to learn on. The larger, the more notes are looked at when predicting
    :param learning_rate: how fast to learn
    :param epochs: how often to improve the loss function
    :param loss_weights: how input variables are weighted
    :param num_files: how many files to take into account of those given
    :param model: if defined, no new model will be trained, but training will continue on the old model
    :return:
    """
    print("Selected Files: ", filenames[:num_files], "Len:", len(filenames[:num_files]))

    # pack all notes of the selected files together
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)
    all_notes = pd.concat(all_notes)  # basically turns it into one big song

    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)  # turn array into key_order

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)  # just another data format
    seq_ds = create_sequences(notes_ds, seq_length)
    for seq, target in seq_ds.take(1):
        print('sequence shape:', seq.shape)
        print('sequence elements (first 10):', seq[0: 10])
        print()
        print('target:', target)

    batch_size = 64
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))
    if not model:
        input_shape = (seq_length, 3)

        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = {
            'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
            'step': tf.keras.layers.Dense(1, name='step')(x),
            'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }

        model = tf.keras.Model(inputs, outputs)

        loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=optimizer)

        model.summary()  # prints model summary

        model.compile(
            loss=loss,
            loss_weights=loss_weights,
            optimizer=optimizer,
        )

    # gw = GetWeights()
    # gradient_callback = GradientPlotCallback()  # doesn't work

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            verbose=1,
            restore_best_weights=True),
        # gw,
    ]
    print("LOSS BEFORE", model.evaluate)
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        # shuffle=True,  # already shuffled before

    )
    plot_loss(history, file_name=FILE_NAME.format(identifier=identifier, name="plot_loss",
                                                  time=time,
                                                  extension="png"))

    model.save(f"{identifier}_{datetime.datetime.now().strftime('%H:%M:%S')}_.model")
    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed = 42
    time = datetime.datetime.now().strftime('%H:%M:%S')
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # filenames = glob.glob(str(maestro / '**/*.mid*'))
    # filenames = get_music_by_emotion("tense")
    seq_length = 50
    for identifier in ["Q1"]:  # , "Q2", "Q3", "Q4"]:
        # filenames = glob.glob(str(MAESTRO / '**/*.mid*'))
        # first_model = train(identifier, filenames, epochs=50, seq_length=seq_length)
        filenames = glob.glob(f"data/EMOPIA_1.0/midis/{identifier}*")
        # do_stuff()
        model = train(identifier, filenames, epochs=1, seq_length=seq_length, num_files=1)
        instrument_name, generated_notes = generate_notes_from_file(model=model, base_file=filenames[0],
                                                                    seq_length=seq_length)
        midi_file = FILE_NAME.format(identifier=identifier,
                                     name="example",
                                     time=time,
                                     extension="midi")
        notes_write_to_midi(
            generated_notes, out_file=midi_file, instrument_name=instrument_name)

        plot_notes(generated_notes, file_name=FILE_NAME.format(identifier=identifier,
                                                               name="plot_notes",
                                                               time=time,
                                                               extension="png"))
        plot_distributions(generated_notes, file_name=FILE_NAME.format(identifier=identifier,
                                                                       name="plot_distributions",
                                                                       time=time,
                                                                       extension="png"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
