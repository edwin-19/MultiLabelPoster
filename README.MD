# Multilabel classification for poster dataset
This repo was nothing more than to show and test out a few concepts using tensorflow 2 on multilabel classification

You can download the poster csv [here](https://www.kaggle.com/neha1703/movie-genre-from-its-poster?select=SampleMoviePosters)
Remember to put it in the dat directory
```python
data/
├── MovieGenre.csv
├── downloaded_posters

```

Command to download images from the selected csv
```python
python download.py
```

# TODO
- [x] download script
- [x] simple eda
- [x] train and evaluate model

# References
- https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72