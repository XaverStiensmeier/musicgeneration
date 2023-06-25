# musicgeneration

## CLI
If no parameter is set, training will start. Training is hardcoded in the script. CLI is only used for using existing models.

- '-m', '--model', type=str; Take a model path. If defined, no model will be trained, but a music piece generated. Requires file.
- '-f', '--file', type=str; File to read input sequence from for music generation
- '-s', '--sequence_length', type=int; Length of sequence to read from file. Should be <= training sequence length
- '-d', '--download', action="store_true"; If set, data set is downloaded.