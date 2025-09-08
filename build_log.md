### 8/24/2025 - Eigen integrated
Turns out Eigen can only do 1 & 2D transforms so I had to "flatten out" the objects that required transformation and work on each dimension separately. 3 days of work.

### 8/25/2025 - Tensor Transformer
Got the transformer code wired in. Some really crazy geometry goes into making machines seem like they're talking to you.

### 8/27/2025 - Lots of Changes
Completly re-worked the cmakefile chain; now there's only one master cmakefile. No more parameters to feed to the root cmake file, invoke normally with 'cmake ..'. BLAS math library now a requirement (Debian: apt get install). The refactor has introduced some serious speed regressions so next coding session will be all about speed optimization.

### 8/30/2025 - Optimization
Optimized the tokenizer and Tensor classes with inline assembly for some of the more time-intensive calculations, more optimizations coming.

### 9/4/2025 – Expanded Tokenization
Spent several days chasing down some funky little errors with the tokenizer while expanding its capabilities (in so doing created some issues with the internationalization code), finally cracked it a few hours ago.

### 9/4/2025 - Conversation and ConversationTurn structures implemented
Put in the foundational structures for getting conversations going on this framework. Also straitened out some lingering issues with the Training class.

### 9/7/2025 - Using Efficient Token Sequence-Based Approach
Hashing the tokens rather than string manipulation is a completely faster approach and I don't even feel the need to use inline assembly. 1000% more
efficient. Added a vectorhash struct to effeiciently manipulate them as well.
