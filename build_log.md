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
Put in the foundational structures for getting conversations going on this framework. Also straitened out some lingering issues with the Training class. Started using the Ceral C++ serialization library, this is automatically downloaded for you while CMake runs.

### 9/7/2025 - Using Efficient Token Sequence-Based Approach
Hashing the tokens rather than string manipulation is a completely faster approach and I don't even feel the need to use inline assembly. 1000% more
efficient. Added a vectorhash struct to effeiciently manipulate them as well.

### 9/9/2025 – Changed my mind about assembly with the Tensor class, removed the now redundant Transformer & LayerNorm classes as they are no longer needed with the for more flexible TransformerModel class.

### 9/10/2025 – Moved the Todos and explanatory papers into their own folder.

### 9/13/2025 – Started on inference engine. Almost as taxing as the Tensor class.

### 9/14/2025 – Moved the tests to their own folder, they’re getting to be to many to simply leave in the src dir.

### 9/16/2025 – Incremental tokenizer/inference improvements, just little dabs here and there.

### 9/17/2025 – Epochal (cyclic) training implemented and seems to be working well. Speed still fairly limited given development is on a linux. Next step is a Window port to I can take advantage of my GPU. I debated several options, including a rebuild of my host pc and install a dual-boot situation, but I really like my rig as it is, so its a windows version of “Monster Baby” (new working title) so as to be able to take advantage of my laptop’s on-board AMD GPU. Run the “starter_convo” to see the training in action.
