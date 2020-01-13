# Federated Learning Tangle Experiment

Use pipenv to set up your environment.

Write the EMNIST dataset to disk using `write_emnist_dataset.py`.

Customize `NUM_ROUNDS` and `NUM_CLIENTS` in `experiment.py` and run it.

For executing a single step (i.e. a particular client training on local data and submitting a transaction), run `step.py f0000_14 10`, where `f0000_14` is the client id and `10` corresponds to `tangle_data/tangle_10.json`.

To view tangles in a web browser, run `python -m http.server` in the repository root and open [http://localhost:8000/viewer/](http://localhost:8000/viewer/). You may need to slide the slider to the left to see something.
