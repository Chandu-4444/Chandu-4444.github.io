---
layout: post
title: "FastAI.jl Textual Development, GSoC 2022"
subtitle: "Blog post summarising the work done during my GSoC 2022"
background: ""
categories: blog gsoc
# use_math: true
---

This summer, I have worked in the development of textual modules in FastAI.jl for GSoC'22 under the NumFOCUS organization. The goal of the project is to develop textual pipelines in FastAI.jl (for text classification and text generation to be specific).

The project is divided into two parts:

1. **Text Clessification Pipeline**: This is the first part of the project, where I have developed a text classification pipeline in FastAI.jl. This pipeline is based on [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/) dataset.
2. **Text Generation Pipeline**: This is the second part of the project, where I have developed a text generation pipeline in FastAI.jl.

The project page can be found [here](https://summerofcode.withgoogle.com/programs/2022/projects/HogSGyea).

## Text Classification Pipeline

The dataset being used here is IMDb large dataset. This contains 50,000 reviews out of which 25,000 were for training and 25,000 are for testing. The dataset is divided into 25,000 positive reviews and 25,000 negative reviews. The dataset is available [here](https://ai.stanford.edu/~amaas/data/sentiment/). The goal of this part of the project is to develop a classification pipeline around this dataset.

### Dataset Recipe

The recipe implemented for IMDb dataset can be used for any other text dataset that has the same folder structure as IMDb. The input is wrapped up in a `Paragraph` block and the output is wrapped up in a `Label` block. `Paragraph` is nothing but a text paragraph.

```julia
julia> using FastAI, FastText
julia> data, blocks = load(datarecipes()["imdb"])
julia> getobs(data, 1), numobs(data)
(("Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific
example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy
chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually
making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make
Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good
cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen
briefly.", "neg"), 25000)
```

### Task

The `TextClassificationSingle` is used to encode the input and output into a format that is acceptable by the model. The preprocessing steps for `Paragraph` and `Label` are as follows:

1. **Paragraph**

   - `Sanitize`: The input is first cleaned using `Sanitize` encoding that cleans the input `Paragraph` by applying `fastai` specific encodings (adding special tokens like xxup, xxmaj, etc), removing punctuations, removing html tags, removing number etc. The output of this encoding is a `Paragraph` (but sanitized).
   - `Tokenize`: The sanitized input is the converted into tokens by leveraging the `TextAnalysis.jl` functions. The output of this enodings is `Tokens`.
   - `EmbedVocabulary`: The tokens are then converted into indices using this encoding. The output of this encoding is `NumberVector`.

2. **Labels**
   - `OneHot`: The labels are converted to one-hot vectors using this encoding.

```julia
julia> task = TextClassificationSingle(blocks, data)
SupervisedTask(Paragraph -> Named{:target, Label{String}})

julia> td = taskdataset(data, task, Training())
julia> td[1]
([35, 3, 68, 7, 6, 130, 41, 52, 7567, 1400  …  11610, 5, 3, 16709, 3, 8050, 56, 33, 113, 3342], Bool[1, 0])
```
