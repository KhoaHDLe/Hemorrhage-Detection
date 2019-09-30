## Hemorrhage-Detection

__Our goal__

Identify acute intracranial hemorrhage and its sub-types. We are asked to predict the occurence and the subtype of intracranial hemorrhage. For this purpose we have to make 6 decisions per image: 5 subtypes and if there is an occurence (any). We can clearly see that we have to make several predictions for one image id:
- Epidural
- Subdural
- Subarachnoid
- Intraparenchymal
- Intraventricular
- Any - this one indicates that at least one subtype is present, hence it tells us if the patient has IH or not.

__Insights__

- The first image already shows that we will have much more zero occurences than positive target values.
- Going into details of each subtype we can see that we have to deal with high class imbalance.
- Epidural is the worst case. For this type we only have a few (< 1%) of positive occurrences. It will be difficult to train a model that - is robust enough and does not tend to overfit.
