# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See https://www.tensorflow.org/tutorials/audio/music_generation
# Temperature https://www.tensorflow.org/text/tutorials/text_generation

import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import os

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
pretty_midi.pretty_midi.MAX_TICK = 1e10
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_music_by_emotion(emotion, toptag=True):  # ONLY FOR YM2413 DATABASE
    df = pd.read_csv('data/YM2413-MDB-v1.0.2/emotion_annotation/verified_annotation_old.csv', delimiter=',')

    # Filter DataFrame based on emotion
    if toptag:
        filtered_data = df[df['toptag_eng_verified'] == emotion]
    else:
        filtered_data = df[df['verified_tags'].str.contains(emotion)]

    # wav to midi
    fnames = [f"data/YM2413-MDB-v1.0.2/midi/adjust_tempo_remove_delayed_inst/{name[:-3]}mid" for name in filtered_data['fname']]
    print([name.split("/")[-1] for name in fnames])
    print(len(fnames))
    return fnames


def notes_to_midi(
        notes: pd.DataFrame,
        out_file: str,
        instrument_name: str,
        velocity: int = 100,  # note loudness
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


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))
    plt.show()


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)
    plt.show()


def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:seconds * _SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)


def save_audio(pm: pretty_midi.PrettyMIDI, name='done.mid'):
    pm.write(name)


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def analyze(file):
    pm = pretty_midi.PrettyMIDI(file)

    # get number of instruments
    print('Number of instruments:', len(pm.instruments))
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)

    # extract notes
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name},'
              f' duration={duration:.4f}')

    # get note data
    raw_notes = midi_to_notes(file)

    # get node names instead of pitches
    get_note_names = np.vectorize(pretty_midi.note_number_to_name)
    sample_note_names = get_note_names(raw_notes['pitch'])
    print(sample_note_names[:20])

    # plot 100 notes
    plot_piano_roll(raw_notes, count=100)
    # plot all notes
    plot_piano_roll(raw_notes)
    # plot node distribution
    print("Distributions")
    plot_distributions(raw_notes)

    # write your own midi
    # example_file = 'example.midi'
    # example_pm = notes_to_midi(
    #    raw_notes, out_file=example_file, instrument_name=instrument_name)
    # display_audio(example_pm)
    # save_audio(example_pm)


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

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten) # now we have sequences starting with one less the further you go

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]  # divide pitch by vocab size
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]  # take everything but last element of a sequence as input
        labels_dense = sequences[-1]  # take last element of a sequence as label
        key_order = ['pitch', 'step', 'duration']
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}
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


def train(identifier):
    num_files = 15
    print("Selected Files: ", filenames[:num_files], "Len:", len(filenames[:num_files]))

    # pack all notes of the selected files together
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)
    all_notes = pd.concat(all_notes)  # basically turns it into one big song

    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)  # turn array into key_order

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)  # just another data format
    seq_length = 25
    vocab_size = 128  # maximum for pretty_midi
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

    input_shape = (seq_length, 3)
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

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
        loss_weights={
            'pitch': 0.05,  # 0.05
            'step': 1.0,
            'duration': 1.0,
        },
        optimizer=optimizer,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]
    epochs = 25
    print("LOSS BEFORE", model.evaluate)
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        # shuffle=True,  # already shuffled before

    )
    print("LOSS AFTER", model.evaluate)
    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()

    # Generate Notes
    temperature = 2.0
    num_predictions = 120
    sample_file = filenames[0]
    raw_notes = midi_to_notes(sample_file)
    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    for x in range(2):
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
            generated_notes, columns=(*key_order, 'start', 'end'))

        get_note_names = np.vectorize(pretty_midi.note_number_to_name)
        sample_note_names = get_note_names(raw_notes['pitch'])
        print(sample_note_names[:10])

        pm = pretty_midi.PrettyMIDI(sample_file)
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        example_file = f"{identifier}_example_{datetime.datetime.now().strftime('%H:%M:%S')}_{x}.midi"
        example_pm = notes_to_midi(
            generated_notes, out_file=example_file, instrument_name=instrument_name)
        save_audio(example_pm, example_file)

        plot_piano_roll(generated_notes)
        plot_distributions(generated_notes)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    maestro = pathlib.Path('data/maestro-v2.0.0')
    if not maestro.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
    # filenames = glob.glob(str(maestro / '**/*.mid*'))
    # filenames = get_music_by_emotion("tense")
    for identifier in ["Q1"]:  # , "Q2", "Q3", "Q4"]:
        filenames = glob.glob(f"data/EMOPIA_1.0/midis/{identifier}*")
        # do_stuff()
        train(identifier)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
