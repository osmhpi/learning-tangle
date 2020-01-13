# Federated Learning Tangle Experiment

Use pipenv to set up your environment.

Write the EMNIST dataset to disk using `write_emnist_dataset.py`.

Customize `NUM_ROUNDS` and `NUM_CLIENTS` in `experiment.py` and run it.

To show tangles in a web browser, run `python -m http.server` in the repository root and open [http://localhost:8000/viewer/](http://localhost:8000/viewer/). You may need to slide the slider to the left to see something.
