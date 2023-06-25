import pretty_midi
import pandas as pd
import collections
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    # See https://www.tensorflow.org/tutorials/audio/music_generation#extract_notes
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
    # See https://www.tensorflow.org/tutorials/audio/music_generation
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
    # plot all notes
    plot_notes(raw_notes)
    # plot node distribution
    plot_distributions(raw_notes)


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5, show=False):
    # See https://www.tensorflow.org/tutorials/audio/music_generation#extract_notes
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))
    if show:
        plt.show()
    else:
        plt.savefig("Plot Distributions")


def plot_notes(notes: pd.DataFrame, count: Optional[int] = None, show=False):
    # See https://www.tensorflow.org/tutorials/audio/music_generation#extract_notes
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
    if show:
        plt.show()
    else:
        plt.savefig()


def plot_loss(history, show=False):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 4, 1)
    plt.title("Total Loss")
    plt.plot(history.epoch, history.history['loss'], label='total loss')

    plt.subplot(1, 4, 2)
    plt.title("Step Loss")
    plt.plot(history.epoch, history.history['step_loss'], label='step loss')

    plt.subplot(1, 4, 3)
    plt.title("Duration Loss")
    plt.plot(history.epoch, history.history['duration_loss'], label='duration loss')

    plt.subplot(1, 4, 4)
    plt.title("Pitch Loss")
    plt.plot(history.epoch, history.history['pitch_loss'], label='pitch loss')

    if show:
        plt.show()
    else:
        plt.savefig("example")
