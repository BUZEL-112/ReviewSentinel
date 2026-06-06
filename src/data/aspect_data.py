"""
Aspect Data Module for Few-Shot Aspect-Based Sentiment Analysis

This module defines the few-shot training examples used by SetFit to learn
aspect categories from minimal labeled data. Examples are manually curated
and do not require aspect tags from the original dataset.
"""

from datasets import Dataset

# ---------------------------------------------------------------
# Manually curated few-shot examples per aspect category
# Add 8-16 examples per aspect for best SetFit performance
# ---------------------------------------------------------------

ASPECT_EXAMPLES = {
    "quality": [
        "This is very well made and feels incredibly sturdy.",
        "The plastic feels extremely cheap and flimsy.",
        "One of the prongs broke off as soon as I took it out.",
        "High quality materials, it has a nice weight to it.",
        "It fell apart the second time I tried to use it.",
        "The build quality is terrible, completely completely fell apart.",
        "Very good quality, must be stainless steel.",
        "The handle literally bent using it for the first time.",
        "It is sturdy and durable, definitely not a cheap knockoff.",
        "The mirror was crooked and the hinges were loose.",
        "Better than stock, highly recommend the craftsmanship.",
        "Poorly constructed and the parts don't align properly.",
        "It's kind of heavy which makes it feel real and premium.",
        "The bristles are super bent and the brush is ruined.",
        "Terrible quality DO NOT BUY!",
        "It feels like it will last a very long time."
    ],
    "shipping": [
        "It arrived a day before the promised date.",
        "Fast shipping, got to my house in two days.",
        "Took over a month to get here, very frustrating.",
        "The delivery was delayed by over a week.",
        "Arrived right on time for my daughter's birthday.",
        "I paid for expedited shipping but it still took ten days.",
        "Quick delivery, exactly as described.",
        "My package was lost in transit and never arrived.",
        "Shipping was way too slow for the price I paid.",
        "It finally showed up weeks after I ordered it.",
        "Arrived more than two weeks sooner than expected.",
        "The shipping carrier delivered it to the wrong address.",
        "I was surprised by how fast it was delivered.",
        "Tracking never updated, but it showed up eventually.",
        "They shipped it out the very same day I ordered.",
        "Terrible shipping experience, it took forever."
    ],
    "price": [
        "Worth every single penny I paid for it.",
        "A complete waste of money, do not buy.",
        "Great value for the price, much cheaper than the salon.",
        "Way too expensive for such a tiny bottle.",
        "Incredible price for the amount of product you get.",
        "Don't throw your money away on this garbage.",
        "It's a steal when you realize how much individual pieces are.",
        "You get what you pay for, it was very cheap.",
        "I feel like this was a total rip off.",
        "Best overall price for this amazing product anywhere.",
        "It is a bit pricey, but the quality justifies it.",
        "Not worth the price, I could have bought this at the dollar store.",
        "Affordably priced and does exactly what I need.",
        "I overpaid for something that doesn't even work.",
        "Great deal, I searched everywhere and this was the lowest cost.",
        "Save your coin and buy a different brand."
    ],
    "customer_service": [
        "The seller was amazing and issued a refund immediately.",
        "Customer service is non-existent, they never replied.",
        "She was consistent with the replies and made sure I was happy.",
        "I contacted the manufacturer and got absolutely no response.",
        "They were very rude when I asked for a replacement.",
        "Fortunately I called customer support and they fixed it right away.",
        "The company reached out to make things right, great support.",
        "They refused to let me return the defective item.",
        "Seller is super nice and even checks in to see how you like it.",
        "I am still waiting for an email back from their support team.",
        "They sent me a fresh batch immediately, no questions asked.",
        "Terrible experience trying to get a hold of a real person.",
        "Customer care was polite, prompt, and very helpful.",
        "I had to file a claim because the seller ignored my messages.",
        "Excellent customer service, they truly care about their buyers.",
        "They won't let you return this so it's a total waste."
    ],
    "packaging": [
        "It was wrapped in so much bubble wrap to avoid any damage.",
        "The package was damaged and open when it arrived on my porch.",
        "One color had leaked all over the inside of the box.",
        "The pump was missing and they put a fake seal on it.",
        "Love the packaging, they packed it with so much care.",
        "The bottle spilled everywhere in transit.",
        "The box was completely crushed but the product was okay.",
        "It came in a beautiful little gift box.",
        "The lid wasn't screwed on tight and it made a mess.",
        "Packaging was secured properly and nothing leaked.",
        "It arrived leaking out of the envelope.",
        "The compact was shattered because there was no padding in the box.",
        "It comes in a convenient case for easy storage.",
        "They shipped a glass bottle in a thin bag, so of course it broke.",
        "The spray nozzle was broken off inside the package.",
        "Beautifully boxed and perfect for gift giving."
    ],
    "scent": [
        "Love this fresh clean scent.",
        "Minus 2 stars for gross scent! Some weird floral scent mix!",
        "Smells like carpet freshener, the scent is overwhelming.",
        "Absolutely no chemical smell at all.",
        "The smell is sooo intense and overpowering.",
        "It has a very nice, very mild musky type scent.",
        "Smells like rotten eggs, absolutely disgusting.",
        "The peppermint is energizing and tingly.",
        "It smells a little like grass initially but fades quickly.",
        "It has a sweet flavor and beautiful coconut aroma.",
        "The fragrance is really unique and reminds me of camping.",
        "I can't get over the terrible smell, I dread using it.",
        "Smells fabulous, I constantly get compliments on it.",
        "There was absolutely no fragrance in the bottle.",
        "It stinks, more like a lady from the 19th century.",
        "The cherry almond scent is my absolute favorite."
    ],
    "effectiveness": [
        "It magically brushes through anything without pulling.",
        "Completely destroyed my blow dryer and made it melt.",
        "It does absolutely nothing to hold my hair in place.",
        "Zapping wrinkles and hydrating thirsty skin instantly.",
        "It removes all of my makeup without leaving any residue.",
        "This product is completely useless and doesn't work.",
        "It alleviated the pockets in my gums just like the dentist said.",
        "My hair felt like straw after using this shampoo.",
        "It took out hair color that other removers couldn't touch.",
        "Did not straighten my hair very well at all.",
        "The best self tanner out there, gives a nice glow.",
        "This does not work as an eyeliner or a shadow.",
        "It plumps up wrinkles and gives a dewy fresh appearance.",
        "It didn't curl my lashes even a little bit.",
        "Works perfectly to separate and store my eye palettes.",
        "Doesn't do the job, it leaves my hair incredibly frizzy."
    ],
    "ease_of_use": [
        "Super easy to use, convenient and good quality.",
        "Had a huge struggle to apply these lashes.",
        "The instructions are minimal and it's hard to set up.",
        "Very easy to apply, basically like a big chapstick.",
        "It takes a lot of effort to remove it.",
        "Easy peasy, I figured it out without reading the manual.",
        "The pump is extremely hard to press down.",
        "Simple and easy to apply for beginners.",
        "It's a bit tricky to get the angle right at first.",
        "They are difficult to use and very messy.",
        "It's very hard to get the tools in and out of the file.",
        "Glides on effortlessly without any tugging.",
        "The stencils make doing my eyebrows completely foolproof.",
        "It's so awkward to hold and drops out of my hand constantly.",
        "Makes sectioning my hair during a blowout a breeze.",
        "Applying this is incredibly time consuming."
    ],
    "longevity": [
        "It goes on easy and lasts forever.",
        "The gold completely faded after 30 minutes.",
        "They stayed on my nails for 15 days without chipping.",
        "It flakes almost instantly, didn't last an hour.",
        "Stays on all day and night through sweat and humidity.",
        "Only lasted about 3 weeks before getting super tangled.",
        "The battery died after a week and won't recharge.",
        "Doesn't last very long, it wears off by lunch time.",
        "It held up well through a whole day of swimming.",
        "The color blurs away within minutes of applying.",
        "I've been using the same set for months and they still look new.",
        "It stopped working after two weeks of use.",
        "The charge lasts longer than my previous toothbrush.",
        "It dried out in the tube after only a month.",
        "This lip tint has incredible staying power.",
        "The motor burned out on the second use."
    ],
    "aesthetics": [
        "It makes my eyebrows look incredibly natural.",
        "Looked like I had a cat on my head.",
        "Gorgeous, deep purple-black cherry color.",
        "Hideous color, it was brown instead of red.",
        "The finish looks really really fake and plastic.",
        "It gives my face a velvety, flawless glow.",
        "They look incredibly dramatic and heavy on my eyes.",
        "The curls are very fancy, nice, and perfectly shaped.",
        "It leaves a weird white cast on my skin.",
        "Very pretty layered necklace, looks expensive.",
        "The wig part looks awful and totally unnatural.",
        "It perfectly matches my skin tone.",
        "The glitter has a beautiful, shiny finish.",
        "It made my hair look dull and lifeless.",
        "The design is super adorable and stylish.",
        "It looks cheap and poorly painted."
    ],
    "texture": [
        "It is a very light, runny, loose cream.",
        "Super soft bristles that feel great on the face.",
        "It was so thick and clumpy, totally unusable.",
        "The lace feels like a hard scratchy plastic.",
        "Very thin and watery shampoo, hard to hold in your hand.",
        "It's extremely greasy and leaves an oily film.",
        "Smooth, velvety consistency that absorbs quickly.",
        "It dries sticky and feels uncomfortable on the skin.",
        "The powder is finely milled and very silky.",
        "It feels like sandpaper rubbing against my face.",
        "The lotion is quite thin and dispenses too fast.",
        "It has a weird, gritty feel to it.",
        "It foams up beautifully into a rich lather.",
        "The gel is incredibly stiff and hard to spread.",
        "Very soft polygel that is easy to manipulate.",
        "It left my hair feeling gummy and weighed down."
    ],
    "physical_reaction": [
        "It doesn't cause any breakouts or clog my pores.",
        "Gave me a terrible rash on my underarms.",
        "I did experience a little itchiness from this hair.",
        "Very gentle on my sensitive skin, no redness at all.",
        "It severely irritated my eyes and made them water.",
        "My skin broke out in hives immediately after applying.",
        "It feels cooling and soothing on my tired muscles.",
        "The metal brush was hurting and pulling my scalp.",
        "It burned my face and left a chemical burn.",
        "Perfect for my baby's delicate skin.",
        "It dried my skin out completely and made it flake.",
        "I had no allergic reaction to this formula.",
        "My gums were bleeding after using this.",
        "It completely relieved the pain in my knee.",
        "The gel stings a bit when you first put it on.",
        "Left my face incredibly soft without irritation."
    ],
    "authenticity": [
        "Definitely a knock off, barely bigger than a chapstick.",
        "This is the real deal, authentic product.",
        "This was not the original product, it's a fake item.",
        "I think these are fake, they don't feel like MAC quality.",
        "It has Greek writing all over it, I'm convinced it's sketchy.",
        "Seems to be the authentic formula I remember from the 90s.",
        "This is a counterfeit, the packaging is totally wrong.",
        "100% genuine, came sealed with the official holographic sticker.",
        "It's a scam, not the perfume shown in the image.",
        "Sadly this is not the same as the original formula.",
        "Verified authentic through the brand's website.",
        "The label looks photocopied, obviously a fake.",
        "I compared it to the one I bought at Sephora and it's real.",
        "It's a cheap imitation from overseas.",
        "Legit product, exactly what I normally buy at the salon.",
        "I believe they changed the formula, this is an imposter."
    ]
}




def get_aspect_labels():
    """Return ordered list of aspect category names."""
    return list(ASPECT_EXAMPLES.keys())


def get_label2id():
    """Map aspect name to integer id."""
    return {label: idx for idx, label in enumerate(get_aspect_labels())}


def get_id2label():
    """Map integer id back to aspect name."""
    return {idx: label for label, idx in get_label2id().items()}


def build_setfit_dataset():
    """
    Convert ASPECT_EXAMPLES into a HuggingFace Dataset
    ready for SetFit training.

    Returns:
        Dataset: HuggingFace dataset with 'text' and 'label' columns
                 where label is the integer id of the aspect.
    
    Raises:
        ValueError: If any aspect category has fewer than 2 examples.
    """
    label2id = get_label2id()

    texts, labels = [], []

    for aspect, examples in ASPECT_EXAMPLES.items():
        if len(examples) < 2:
            raise ValueError(
                f"Aspect '{aspect}' has {len(examples)} example(s). "
                f"SetFit requires at least 2 — recommended 8-16."
            )
        for example in examples:
            texts.append(example)
            labels.append(label2id[aspect])

    return Dataset.from_dict({"text": texts, "label": labels})


def validate_examples():
    """
    Print a summary of how many examples exist per aspect.
    Call this before training to catch empty or thin categories.
    """
    print("Aspect Example Counts:")
    print("-" * 30)
    for aspect, examples in ASPECT_EXAMPLES.items():
        status = "OK" if len(examples) >= 8 else "LOW"
        print(f"  {aspect:<20} {len(examples):>3} examples  [{status}]")
    print("-" * 30)