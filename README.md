# Heron Inference

Among the most typical inference methods for probabilistic models, variational inference tends to approach but not reach the predictive power of sampling approaches. On the other hand, Gibbs sampling methods are inherently sequential and hard to adapt to the required computational efficiency of the big data era. 

Heron presents a new inferential approach derived from the Gibbs-based methods which is able to improve both the predictive power and computational efficiency of the typical inferential methods. The code implements the paper [Heron Inference for Bayesian Graphical Models](https://arxiv.org/abs/1802.06526)


### Benefits:
* **Convergence Assesment**
Heron method maximizes the state augmentation of a topic model and finds a deterministic approach to Gibbs sampling. The difficulty in the assesment of sampling methods stems from having to estimating parameters based on samples. With the deterministic approach, the assessment of convergence is done directly from the parameters of the model. 

* **Predictive Power**
The augmentation of the latent state is known to increase the predictive power of probabilistic models, by taking this augmentation to infinity, Heron improves the learning of the probabilistic models.

* **Transformation of the inference problem to a well studied field**
By studying the convergence properties of the deterministic algorithm. We find that the Gibbs sampler is ultimately solving a non-linear system of equations. Hence, we can use fixed-point iterative methods to infer the parameters of topic models

* **Theoretical distributabuility**
As a benefit from the fixed-point iterative sovler, Heron can be distributed into as many cores as the number of data points. Independently of the number of cores, the distributed algorithm always converges to the same exact result.

* **Computational efficiency**
Given that the method is distributable and deterministic. GPU implementation is favorable and the bottleneck imposed by the availability of hardware-based samplers inside the GPU is no longer an issue. 


## Code:

### Datasets
We provide a small dataset for fast testing of the code. 

| Dataset   |  LDA    | SLDA   | RTM | 
| ------------- | -------------|  ------------- |------------- |
| Cora   |  :heavy_check_mark: | | :heavy_check_mark:  |
| Movielens |  :heavy_check_mark: | :heavy_check_mark:  | |


### Inference methods

| Method   |   CPU    | GPU   | 
| ------------- | -------------|  ------------- |
| CGS   |  :heavy_check_mark: | | 
| Cool <sup>[1](https://arxiv.org/abs/1409.5402)</sup>|  | :heavy_check_mark: |
| Heron  | :heavy_check_mark: | :heavy_check_mark: |

<sup>[1](https://arxiv.org/abs/1409.5402)</sup> Algorithm that cools the posterior by doing a finite augmentation of the latent state. 

### Probabilistic Models

| Model | CGS | SAME  | Heron |
| ------------- | -------------|  ------------- |------------- |
| LDA    |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | 
| SLDA |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | 
| RTM    |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | 

### Prerequisites:

* Numpy  (>=1.13)
* [Pycuda](https://documen.tician.de/pycuda) (>=2017.1)<sup>2</sup>
* [Cuda](http://www.pradeepadiga.me/blog/2017/03/22/installing-cuda-toolkit-8-0-on-ubuntu-16-04) (>=8.0)<sup>2</sup>

<sup>2</sup> For GPU versions

### Learning:

The following script is used to learn a model using a selected inference method. We provide a sample training to allow our code to work out of the box. 

Usage: main.py [Options] 

`python main.py -f Datasets/cora/train.npy -m RTM --inference herongpu --alpha 0.75 --beta 0.75 --eta 1.0 -k 20 -i 100 --path Save/`

`python main.py -f Datasets/movielens/dictionateddata.npy -m SLDA --inference herongpu --alpha 0.75 --beta 0.75 --eta 1.0 -k 20 -i 100 --path Save/`


| Options:        | Description           | 
| ------------- | ------------- | 
| -h, --help            | show this help message and exit |
|  -f FILENAME          | Path to the filename of the dataset|
|  -m MODEL             | LDA, RTM or SLDA [default: LDA]| 
|  --inference=INFERENCE| cgs, heron, cool or herongpu [default: heron]
|  --batch=BATCH        | Number of tuples in a batch. 0 indicates that no batches are performed which is the fastest version if the RAM can cope. Note that the CPU versions, namely cgs and heron do not support batches. [default: 0]|
|  --alpha=ALPHA        | Hyper-parameter alpha [default: 0.75]|
|  --beta=BETA          | Hyper-parameter beta [default: 0.75]|
|  --eta=ETA            | Hyper-parameter eta [default: 1.0]|
|  -a A                 | Hyper-parameter a [default: 0.5]|
|  -k K                 | Number of topics [default: 20]|
|  -i ITERATION         | Number of iterations [default: 500]|
|  -r RANDOMNESS        | Random initialization: 0 for using a given initialization, or 1 for uniform initialization of the topic assignments [default: 0]|
|  --path=PATH          | path for saving results. if given an empty string "", the parameters will not be saved. [default: Save/]|
|  --seed=SEED          | random seed |
|  --compression        | Experimental support for alternative reading of a different input format.|


### Evaluation:

To facilitate evaluation of the predictive power, we provide the following script:

Usage: evaluation.py [Options] 

`python evaluation.py -d Datasets/cora/ -m RTM -i 100 `

| Options:        | Description      | 
| ------------- | -------------| 
|  -h --help            |show this help message and exit|
|  -d DATASET            |Path to the directory where the results are stored|
|  -m MODEL              |RTM, LDA or SLDA [default: LDA]|
|  --inference=INFERENCE |cgs, heron or cool[default: heron]|
| --alpha=ALPHA          |Hyper-parameter alpha [default: 0.75]|
|  --beta=BETA           | Hyper-parameter beta [default: 0.75]|
|  --eta=ETA             | Hyper-parameter eta [default: 1.0]|
|  -k K                  | Number of topics [default: 20]|
|  -i ITERATION          | Number of iterations [default: 500]|
| --path=PATH            | Path for saving results. Use an empty string "" to avoid saving the parameters. [default: Save/]|

