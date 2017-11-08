# 6.867 Final Project

## Requirements
- Numpy
- NLTK packages:
```
$ sudo pip install -U nltk
$ python
> import nltk
> nltk.download('punkt')
> nltk.download('averaged_perceptron_tagger')
> nltk.download('stopwords')
```

## Examples
Load review data:
```
from get_yelp_data import *
data = get_review_data('../data/review.csv')
```
