# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
    'pitch': 0.05,  # 0.05
    'step': 1.0,  # 1.0
    'duration': 1.0,  # 1.0
}
KEY_ORDER = ['pitch', 'step', 'duration']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_dataset_existence():
    maestro = pathlib.Path('data/maestro-v2.0.0')
    if not maestro.exists():
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


def notes_to_midi(
        notes: pd.DataFrame,
        out_file: str,
        instrument_name: str,
        velocity: int = 100,  # note loudness; currently ignored during training. See 
) -> pretty_midi.PrettyMIDI:
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
    pm.write(out_file)
    return pm


def create_sequences(
        dataset: tf.data.Dataset,
        seq_length: int,
        vocab_size=128,
) -> tf.data.Dataset:
    seq_length = seq_length + 1  # for labels

    # See https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window
    # creates windows of seq_length where the first element is the second element in the element before
    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)  # first element of a line is k*shift; following elem is +stride

    # `flat_map` flattens the" dataset of datasets" into batches of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)  # now we have sequences starting with one less the further you go

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]  # divide pitch by vocab size
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]  # take everything but last element of a sequence as input
        labels_dense = sequences[-1]  # take last element of a sequence as label
        labels = {key: labels_dense[i] for i, key in enumerate(KEY_ORDER)}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure_strength = 15
    positive_pressure = positive_pressure_strength * tf.maximum(-y_pred, 0.0)
    # it's called reduce because it reduces multiple values to one by applying mean
    return tf.reduce_mean(mse + positive_pressure)


def predict_next_note(
        notes: np.ndarray,
        keras_model: tf.keras.Model,
        temperature: float = 1.0) -> (int, float, float):
    """Generates a note IDs using a trained sequence model."""

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

    # `step` and `duration` values should be non-negative
    if step < 0:
        print(f"Step <0 during prediction!")
    step = tf.maximum(0, step)
    if duration < 0:
        print(f"Duration <0 during prediction!")
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def generate_notes(model, input_notes, instrument_name, temperature=2.0, num_predictions=240, file_name=None):
    # Generate Notes

    # The initial sequence of notes; pitch is normalized similar to training sequences

    for x in range(1):
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

        example_file = f"{identifier}_example_{datetime.datetime.now().strftime('%H:%M:%S')}_{x}.midi"
        example_pm = notes_to_midi(
            generated_notes, out_file=example_file, instrument_name=instrument_name)
        example_pm.write(example_file)

        plot_notes(generated_notes)
        plot_distributions(generated_notes)


def train(identifier, filenames, seq_length=25, learning_rate=0.005, epochs=25, loss_weights=LOSS_WEIGHTS,
          num_files=15, model=None):
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
    vocab_size = 128  # 128 is maximum pitch range for pretty_midi
    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
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
            patience=5,
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
    plot_loss(history)

    model.save(f"{identifier}_{datetime.datetime.now().strftime('%H:%M:%S')}_.model")

    base_file = filenames[0]
    print("Base File: " + base_file)
    raw_notes = midi_to_notes(base_file)
    pm = pretty_midi.PrettyMIDI(base_file)
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    sample_notes = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)
    generate_notes(model=model,
                   input_notes=(sample_notes[:seq_length] / np.array([vocab_size, 1, 1])),
                   instrument_name=instrument_name)
    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    filenames = []
    # filenames = glob.glob(str(maestro / '**/*.mid*'))
    # filenames = get_music_by_emotion("tense")
    for identifier in ["Q3"]:  # , "Q2", "Q3", "Q4"]:
        filenames += glob.glob(f"data/EMOPIA_1.0/midis/{identifier}*")
        # do_stuff()
        first_model = train(identifier, filenames, epochs=50)
        # filenames = glob.glob(str(maestro / '**/*.mid*'))
        # second_model = train(identifier, filenames[:2])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
