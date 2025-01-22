# Glaze
* Link: https://glaze.cs.uchicago.edu/faq.html
* Glaze is a tool that cloaks online images so AI image generators cannot replicate its style.
* Glaze is image specific. The cloaks necessary to prevent AI from stealing its style vary between images. Glaze's cloaking tool is run locally on the user's device, and "calculates" the cloak needed given the original image and the target style specified.
* Glaze is effective against several different AI models. Once a cloak is added to an image, the same one can keep separate AI models (Midjourney, DallE) from stealing its style. This is called _transferability_. While it's difficult to predict performance on newer models, the tool has been tested on multiple existing ones.
* Glaze cloaks cannot be easily removed from the artwork (like via sharpening, blurring, denoising, downsampling, stripping of metadata, etc.)
* Strong cloaks lead to stronger protection. The tool controls how much a cloak modifies the original artwork (from imperceptible changes to visible ones). Larger modifications provide stronger protection against AI's ability to steal the style.
* Glaze only runs on JPG or PNG files.
## How Does Glaze Work Against AI?
* The Achilles' heel for AI models is that they are unable to approximate what humans see. This is demonstrated in a phenomenon called adversarial examples (small modifications to inputs can result in massive differences in how AI models classify input). It is extremely hard to remove the possibility of adversarial examples from being used in training datasets, and in a way are a fundamental consequence of the imperfect training of AIs.
* The underlying techniques used in the Glaze cloaking tool are similar to those present in adversarial samples, small changes have a massive impact on classification.
## Can you take a Screenshot of an Image to Destroy the Cloak?
* Cloaks make calculated changes to pixels within an image. The changes vary for each image, and while they are not very noticeable to humans, they distort images for AI during the training process, and a screenshot would maintain that.
## Can't a Filter, Compression, Blurring, or Noise be Applied to the Image to Destroy Cloaks?
* No tools work to destroy image cloaks. Cloaking doesn't use high intensity pixels or patterns to distort an image. It is a precisely computed combination of a number of pixels that don't necessarily stand out to the human eye, but can distort the Ai's "vision". It is not a watermark that is seen or unseen, it transforms the image in a dimension that humans do not perceive, but very much in the dimensions that the deep learning model perceive these images. Transformations that rotate, blur, change resolution, crop, etc do not affect the cloak.
## Does Glaze Protect Against Image2Image Transformations?
* Glaze was designed to prevent art style mimicry, but it could be an intense distruption of img2img attacks. Glaze provides some protection against weaker img2img style transfers (like built in functions in Stable Diffusion) but controlnet protection requires much higher intensity settings.
* The only tool that claims strong disruption of img2img attacks is Mist, which has an intensity 5x stronger than Glaze's highest intensity.
## How is Glazing Useful?
* Glazing shifts an AI model's view of one's style in its "feature space" (the conceptual space where AI models interpret artistic styles). Because AI models are always adding more training data in order to improve their accuracy and keep up with trend changes, the more cloaked images you post online, the more your style will shigt in the AI model's feature space, shifting closer to the target style (EX: cubism) and generate images in that style when asked for your style.

# Nightshade
* Link: https://nightshade.cs.uchicago.edu/whatis.html
* Many content creators wish to prevent generative AI models from using their work for training. Opt-out lists can easily be ignored by model trainers (as they are near impossible to verify or enforce), and those who violate do-not-scrape directives cannot be easily or confidently identified.
* Nightshade is a tool developed by the University of Chicago that turns any images into "poison" samples, so models training on them without consent will learn unpredictable, abnormal behaviors (like generating an image of a handbag when an image of a cow is requested). Used responsibly, Nightshade can deter model trainers who disregard copyrights and do-not-scrape/robots.txt directives. It does not rely on the kindness of model trainers, but instead associates a small incremental price on each piece of data scraped and trained without authorization. The goal is to make training on unlicensed data costly enough that licensing images from creators becomes common.
* While Glaze is a defense against mimicry, Nightshade is an offense tool meant to distort feature representations inside of generative AI models. Nightshade is computed as a multi-objective optimization that minimizes visible changes to the original image. Human eyes might see a shaded cow in a field, but the AI model might see a large purse laying in the grass, and if it trains on enough of these images, it will become convinced cows look like purses.
* Nightshade effects are robust to normal changes one might apply to an image (cropping, resampling, compressing, smoothing out pixels, adding noise, taking screenshots)
* Because Glaze is a defensive tool, it should be used on every piece of artwork an artist posts online. Nightshade is an entirely optional feature meant to deter and disrupt model trainers, possibly helping other artists in the future.
## Risks and Limitations
1. Changes made by Nightshade are more visible on art with flat colors and smooth backgrounds. Because Nightshade is about disrupting models, lower levels of intensity/poison do not have negative consequences for the image owner. A low intensity setting can preserve the visual quality of the original image.
2. As with any security attack or defense, Nightshade is unlikely to stay future proof over long periods of time. As an attack, Nightshade can evolve and keep pace with countermeasures/defenses.

# DeepSeek
* Competitor to OpenAI, DeepSeek was formed as an artificial intelligence body of the Chinese company High-Filyer.
* After releasing DeepSeek-V2 in May 2024, which offerred strong performance for cheap, DeepSeek caused an AI model price war in China, causing other companies to cut cost. Currently DeepSeek is purely a research body with no plans for commercialization.
* DeepSeek LLM was released on November 29, 2023, which competed in performance with GPT-4. However, it struggled with both efficiency and scalability. DeepSeek-V2 on release was reported as cheaper than its peers. In November 2024, DeepSeek R1-Lite-Preview was released and designed to excel in tasks like mathematical reasoning and real-time problem solving (but was outperformed by OpenAI o1).
* On January 20, 2025, DeepSeek-R1 and DeepSeek-R1-Zero were released, based on V3-Base. "DeepSeek-R1-Distill" models were not basedon R1, but instead, similar to other open models like Llama, were finetuned on synthetic data generated by R1.
* R1-Zero is solely trained on reinforcement learning, without supervised fine-tuning. It is trained using group relative policy optimization, which estimates the baseline from group scores instead of using a critic model.
