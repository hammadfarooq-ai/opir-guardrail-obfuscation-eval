Hi Ihor, thanks for sharing the checkpoint. I tested both Opir large and edge on 100 harmful + 100 benign JBB prompts, with four synthetic obfuscations.

Zero-width insertion reduced harmful-prompt recall from 97/100 for both models to 1/100 for large and 8/100 for edge. Normalization restored it to 97/100, but also restored the original benign blocking rates: 54/100 and 71/100 under JBB’s labels. On a separate 30-prompt XSTest safe subset, those counts were 0/30 and 4/30.

The main limitation is that my normalization rules mirror the transformations I generated, so this demonstrates recovery on known transforms rather than robustness to unseen attacks. I also can’t rule out training-data overlap.

Code, settings and results: [repository link]

Are these benign blocking rates expected with single-label classification using ["safe", "unsafe"], or would you recommend a different configuration?
