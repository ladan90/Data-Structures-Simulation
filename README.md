This repository provides the Python implementation for the experiments detailed in the paper "Towards Efficient Data Structures for Approximate Search with Range Queries".
The research introduces the c-DAG (Directed Acyclic Graph), a novel, tunable data structure designed to enhance Single-Range-Cover (SRC) approximate range queries. Our findings demonstrate that the c-DAG provably decreases the average number of false positives by a logarithmic factor compared to the classic 1D-Tree, while maintaining asymptotically similar time and memory complexities.

Abstract
Range queries are a fundamental operation in data retrieval, but achieving exact results can be costly. Approximate search offers a faster alternative but can introduce false positives. This work presents the c-DAG, a data-dependent structure that extends the 1D-Tree with tunable, overlapping branches ($c \geq 3$). Through a competitive analysis using a stochastic Level Difference Distribution (LDD) framework, we prove that the c-DAG incurs only a small additive constant search time overhead ($\leq 2 \cdot \frac{c-2}{c-1}$) while achieving a multiplicative logarithmic reduction in the false positive ratio ($\Omega(\log(N/s))$). These theoretical results are extended to empirical distributions using a lightweight machine learning framework and validated on real-world datasets (Gowalla and NYC Yellow Taxi).


