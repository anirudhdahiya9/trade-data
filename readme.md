# Data Processing Readme

## Structure
`./*.ipynb` the notebooks used to preprocess and extract the datasets for each corresponding category.

`./data/*.txt` are the corresponding clean data files.

## Categorywise Remarks

### Location
Used the UN location code data from [here](http://www.unece.org/cefact/locode/welcome.html). Cleaned the country and province names in the notebook. Data currently segregated into cities, subdivisions and countries list.
* Should I consider company addresses as data points too here? Access to that data exists as part of [Corpwatch](https://old.datahub.io/dataset/corpwatch) company registries data. 

###  Company Names
Data fetched from [Corpwatch](https://old.datahub.io/dataset/corpwatch).
Huge collections of company registrations mainly based in the US.
* Generated *1,083,055* unique company names with probabilistic case variations (uppercase, lowercase, mixedcase), as initial data had all company names in ALLCAPS.
* Also generated randomized pruning of suffixes to names, like `& co`, `LLC` etc 
* Example: `TATA SONS & CO` -> `Tata Sons & Co` `tata sons & co` `tata sons` and all case variations.
	**Clarification :** *Ideally the data should include foreign names of companies too, since the data is meant for international trade. Please clarify if current data is alright.*

 ### Products
  Data sourced from [UNCPC product classification data](https://unstats.un.org/unsd/classifications/Econ/CPC.cshtml) (1058 products), and [Flipkart products dataset](https://www.kaggle.com/PromptCloudHQ/flipkart-products) (an ecommerce site in India) (4061 products).
 ##### Preprocessing
 * Flipkart Dataset
	 * Navigated the Product category tree to fetch non-terminal product categories, stemmed the product categories using NLTK wordnet stemmer to generate singular and plural variations, also created case variations. Both processes randomized using random.random().
* UNSTATS CPC
	* Since lots of overlapping and finer product categories, I first create a set of unique items.
	* Since many products are basically descriptions, I take only products upto length 5 words, and also ignore any further specification, like `Rice, seed`. Can create further case variations.

### Date
Collected from [OntoNotesv5](https://catalog.ldc.upenn.edu/LDC2013T19) Dataset, where the NER labels also specify any date expressions. Was able to filter out 23,799 various date expressions from the dataset.
 * *Clarification* : Currently the mined date expressions are more natural language text in nature. Should we focus more on standardized date expressions? like `20-08-2014` and `20th Jan 1948`
 For example, 
	```text
	December 20
	January 1
	December 18
	December 25
	the 21st century
	June of 2001
	December 28
	2001
	Four years ago
	year
	1996
	```
 * Also generated random date data in various formats, check `data/dates_randomgen.txt`	

### Random Strings
This category seems to be meant to capture random ID numbers, like invoice/tracking numbers etc. Created a random number generator, where the delimiter, the number and types of id segments, and the individual segment lengths, all are generated on the fly and randomly. Refer to `random_string.ipynb` for details. Currently generated strings look like
```text
QBFGNVRYQ#cEqufnXwl#wcciamp
IMXHAMZHR#mialgrw
LTTKHPL
8988-AFSDRTDM-EZVWOUHB
cJjeGRbI
721888479
qlbd
vdo-7000
72241-913
6116@7440@EpFfWdpsV
wRY
GLA@KQRNBJZPX
37081-TJUHVUCRL
73263
ECWVPJK:fvcdwmnno
YNOFSLD-7778
86-23
```

### Other category
Not very clear about it, seems like a general category meant for natural language(non ID numbers), but which don't belong elsewhere as well. I took the U.S. Security and Exchange Commission (SEC) filings data from [here](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/SEC-filings). It is annotated with basic named entities, and I capture the non-entity textual spans. Current data looks like
```text
AND
27 , 1999 , between
Bank "), a
principal place
of business at 3003 Tasman
California 95054 with
a loan production office
. 350 , Wellesley
, INC .
. (" Borrower "), whose address
4th Floor , Cambridge ,
Floor
the terms
terms
on which
and Borrower will repay Bank
. The parties agree
```

