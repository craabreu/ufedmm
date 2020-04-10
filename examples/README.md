Examples
========

Alanine-Dipeptide
-----------------

The force field used for the dipeptide is `AMBER ff14SB`.

To simulate a molecule of alanine dipeptide in a vacuum:

```
python alanine_dipeptide.py
```

To simulate a molecule of alanine dipeptide in explicit water:

```
python alanine_dipeptide.py --water tip3p
```

Note: other available water models are `spce`, `tip4pew`, and `tip5p`.
