irt-em
======

Estimate 2PLM IRT by EM algorithm

Example
=======
See the result of 
> ./irt-em < data/jag_2014_summer_day4.txt

This data is generated from a result of a programming contest (http://jag2014summer-day4.contest.atcoder.jp/standings).

Data Format
===========
First line consists of 3 integers
> (a number of responses) (a number of examinees) (a number of items)

Next (a number of responses) lines contains response information.
Each line consists of 3 integers which relates single response.
> (index of examinee) (index of item) (response (positive: 1, negative: 0)
