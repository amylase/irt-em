irt-em
======

estimate 2PLM IRT by EM algorithm

Example
=======
see
./irt-em < data/jag_2014_summer_day4.txt

this data is generated from a result of a programming contest (http://jag2014summer-day4.contest.atcoder.jp/standings).

Data Format
===========
first line consists of 3 integers
(a number of responses) (a number of examinees) (a number of items)

next (a number of responses) lines contains response information.
each line consists of 3 integers which relates single response.
(index of examinee) (index of item) (response (positive: 1, negative: 0)
