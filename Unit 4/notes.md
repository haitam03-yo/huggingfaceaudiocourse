
# Unit 4. Build a Music Genre Classifier

## Pre-trained Models and Datasets for Audio Classification

In classification, we use an encoder-only architecture. The decoder overcomplicates the task, and the same applies to an encoder-decoder architecture. To use a pre-trained model, we need to ensure that the model available in the hub has been trained with the same number of classes we want our model to detect. We then fine-tune it. There are multiple classification tasks.

## Keyword Spotting

Detecting specific keywords in an audio clip.

### Speech Commands

Identifying which command is spoken in the audio.

## Language Identification (LID)

Determining which language is spoken.

## Zero-Shot Audio Classification

A common limitation of classification models is their inability to generalize to more labels. For example, if a model is trained to identify 101 languages, it will not be able to recognize 19 additional languages if we later want to expand it to 120 languages. This makes traditional classification models ineffective for transfer learning.

Zero-shot classification allows the model to detect new, unseen classes using text labels. An example from Hugging Face is the CLAP model, which takes both audio and text (labels) and computes a similarity score between the audio and the given labels.

However, to improve accuracy, we need to provide a subset of relevant labels instead of all possible labels, as an exhaustive label set increases error rates.

So why don’t we always use zero-shot models? The main reason is that they are still trained on specific datasets, limiting their generalization. For instance, CLAP is trained on the ESC dataset, so it can distinguish whether an audio clip contains speech or not, but it cannot determine which language is spoken.
