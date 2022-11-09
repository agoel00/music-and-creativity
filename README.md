# Music and Creativity

Investigating the relationship between music and (verbal) creativity across two dimensions - divergent and convergent.

## Installation

1. Clone this code:

    - `git clone https://github.com/agoel00/music-and-creativity.git`

2. Install [Python 3](https://www.python.org) and [pip](https://pypi.org/project/pip/).

3. Download the dependencies and model:

    - `pip3 install --user numpy scipy`

    - Download and extract glove.840B.300d.zip from <https://nlp.stanford.edu/projects/glove/>

4. Try it:

    - `python3 run.py`

## Examples

```python
import dat

# GloVe model from https://nlp.stanford.edu/projects/glove/
model = dat.Model("glove.840B.300d.txt", "words.txt")

# Compound words are translated into words found in the model
print(model.validate("cul de sac")) # cul-de-sac

# Compute the cosine distance between 2 words (0 to 2)
print(model.distance("cat", "dog")) # 0.1983
print(model.distance("cat", "thimble")) # 0.8787

# Compute the DAT score between 2 words (average cosine distance * 100)
print(model.dat(["cat", "dog"], 2)) # 19.83
print(model.dat(["cat", "thimble"], 2)) # 87.87

# Word examples (Figure 1 in paper)
low = ["arm", "eyes", "feet", "hand", "head", "leg", "body"]
average = ["bag", "bee", "burger", "feast", "office", "shoes", "tree"]
high = ["hippo", "jumper", "machinery", "prickle", "tickets", "tomato", "violin"]

# Compute the DAT score (transformed average cosine distance of first 7 valid words)
print(model.dat(low)) # 50
print(model.dat(average)) # 78
print(model.dat(high)) # 95

# Compute the RAT score
print(model.rat(low))
```


## Credits

This repo is a modification of the original code by [Jay Olson](https://www.jayolson.org) at Harvard University.

The dictionary (words.txt) is based on [Hunspell](https://hunspell.github.io)
by László Németh.
