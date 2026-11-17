# Kohonen Self-Organizing Map (SOM): Psychophysics Classification

Implementation of a Kohonen SOM in NumPy, applied to classifying subjects as
healthy controls or patients from psychophysics marker data.

## Files

| File | Description |
|------|-------------|
| `kohonenSOM01.py` | Small SOM (2 neurons, 1D) trained on toy 4-dimensional vectors (Class I / II) |
| `trainKohonen02.py` | 6x6 SOM trained on the healthy + patient data, with cluster analysis (BMUs used by each group) |
| `Agbani_Miracle_Ibifubara.py` | Full project: trains a 5x5 SOM, classifies the 4 test subjects, and compares SOM to K-means and kNN |

## Data

The scripts expect these whitespace-separated text files in the same folder:
`healthy.txt`, `patient.txt`, `agbani.txt` (test subjects).
They are not included in this repository.

## Run

```bash
pip install -r requirements.txt
python kohonenSOM01.py
python trainKohonen02.py
python Agbani_Miracle_Ibifubara.py
```

## Method

- Weights initialised randomly (small values), fixed seed for reproducibility
- Best Matching Unit (BMU) found by Euclidean distance
- Learning rate and neighbourhood radius decay exponentially with the epoch
- Gaussian neighbourhood function on a 2D grid

## Author

Agbani Miracle Ibifubara
