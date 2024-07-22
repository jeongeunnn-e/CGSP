# CGSP(Cross-Domain Recommendation framework based on Graph Signal Processing)

This is code for CGSP(Cross-Domain Recommendation framework based on Graph Signal Processing).

Also this code includes 3 baselines of GSP-based methods(i.e, GF-CF, PGSP, LGCN-IDE) for unified domain.


## Datasets
We have used two datasets : [Amazon](http://jmcauley.ucsd.edu/data/amazon/index_2014.html) and [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information?resource=download).

Specifically, we have used Amazon Movie and Music, Amazon Sports and Clothes, Douban Movie and Music, Douban Movie and Book.

More details, in cgsp/data/readme.md


## How to Run

To run the code, you can execute the following command in your terminal:

- Parameter Configurations
- test_mode : `src` for inter-domain recommendation, `tgt` for intra-domain recommendation
- simple_model : cgsp-io, cgsp-oa, cgsp-ua
- a : hyperparameter alpha ( recommend 0.85 )

```bash
python main.py --dataset=<dataset name> --dtype=<src/tgt domain> --simple_model=<model name> --a=<alpha> --test_mode=<src/tgt>
python main.py --dataset=amazon --dtype=movie_music --simple_model=cgsp-io --a=0.85
python main.py --dataset=douban --dtype=movie_book --simple_model=cgsp-oa --a=0.85
python main.py --dataset=amazon --dtype=sport_cloth --simple_model=cgsp-ua --a=0.85

```

- Also, you can run GSP-based baselines(i.e, GF-CF, LGCN-IDE, PGSP) in a unified domain.
    - simple_model : gf-cf, pgsp, lgcn-ide
    - test_mode : `merge-src` for inter-domain recommendation, `merge-tgt` for intra-domain recommendation
    - For example,
      
 ```bash
python main.py --dataset=douban --dtype=movie_book --simple_model=gf-cf --test_mode=merge-tgt


```
