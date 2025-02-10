# Unit 2. A gentle introduction to audio applications

Some audio task examples:

* **Audio classification** : easily categorize audio clips into different categories. You can identify whether a recording is of a barking dog or a meowing cat, or what music genre a song belongs to.
* **Automatic speech recognition** : transform audio clips into text by transcribing them automatically. You can get a text representation of a recording of someone speaking, like “How are you doing today?“. Rather useful for note taking!
* **🔹Speaker diarization** : Ever wondered who’s speaking in a recording? With 🤗 Transformers, you can identify which speaker is talking at any given time in an audio clip. Imagine being able to differentiate between “Alice” and “Bob” in a recording of them having a conversation.
* **Text to speech** : create a narrated version of a text that can be used to produce an audio book, help with accessibility, or give a voice to an NPC in a game. With 🤗 Transformers, you can easily do that!

# **Audio Classification with a Hugging Face Pipeline**

We explored how to use Hugging Face's `pipeline` to apply a **pretrained model** for  **audio classification** . This approach worked well because we found a model that was already fine-tuned on a dataset similar to ours.

However, in real-world scenarios, pretrained models may not always perform optimally on our specific data. To achieve better accuracy, we often need to **fine-tune the model** on our own dataset to adapt it to our specific use case.

# {

# 
    Automatic speech recognition (ASR) with a pipeline

# 
    Audio generation with a pipeline

# }

Same as the last chapter.

When working on solving your own task, starting with a simple pipeline like the ones we’ve shown in this unit is a valuable tool that offers several benefits:

* a pre-trained model may exist that already solves your task really well, saving you plenty of time
* pipeline() takes care of all the pre/post-processing for you, so you don’t have to worry about getting the data into the right format for a model
* if the result isn’t ideal, this still gives you a quick baseline for future fine-tuning
* once you fine-tune a model on your custom data and share it on Hub, the whole community will be able to use it quickly and effortlessly via the `pipeline()` method making AI more accessible.
