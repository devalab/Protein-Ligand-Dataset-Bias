# DeepDTA

The [DeepDTA](https://doi.org/10.1093/bioinformatics/bty593) experiments have been conducted by using the open source library [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose).

The ligand only and protein only tests can be conducted by simply changing the following lines in `DeepPurpose/DeepPurpose/DTI.py`

## For clustered cross validation

We can directly load the .pkl files created for davis and KIBA splits into the respective notebook [like for KIBA](https://github.com/kexinhuang12345/DeepPurpose/blob/master/DEMO/DeepDTA_Reproduce_KIBA.ipynb) and make two folds as train and the remining fold as testing.

## For ligand only

convert
`dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]` to `dims = [self.input_dim_drug] + self.hidden_dims + [1]`
and do the following change in forward function of the classifer module.
`v_f = torch.cat((v_D, v_P), 1)` to `v_f = v_D`

## For protein only

convert
`dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]` to `dims = [self.input_dim_protein] + self.hidden_dims + [1]`
and do the following change in forward function of the classifer module.
`v_f = torch.cat((v_D, v_P), 1)` to `v_f = v_P`
