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

__Definition__

Window - How many Houndsfield units within 256 shades of grey
Level - Where is the window is centered

__Definition__

- that hounsfield units are a measurement to describe radiodensity.
- different tissues have different HUs.
- Our eye can only detect ~6% change in greyscale (16 shades of grey).
- Given 2000 HU of one image (-1000 to 1000), this means that 1 greyscale covers 8 HUs.
- Consequently there can happen a change of 120 HUs unit our eye is able to detect an intensity change in the image.
- The example of a hemorrhage in the brain shows relevant HUs in the range of 8-70. We won't be able to see important changes in the 
- intensity to detect the hemorrhage.
- This is the reason why we have to focus 256 shades of grey into a small range/window of HU units. (WINDOW)
- The level means where this window is centered.



