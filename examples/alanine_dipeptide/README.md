Examples
========

Alanine Dipeptide
-----------------

The default force field used for the dipeptide is `AMBER-03`.

To simulate a molecule of alanine dipeptide in a vacuum:

```
python alanine_dipeptide.py
```

To perform post-processing:

```
python analysis.py
``` 

To simulate a molecule of alanine dipeptide in explicit water:

```
python alanine_dipeptide.py --water tip3p
```

Note: other available water models are `spce`, `tip4pew`, and `tip5p`.

