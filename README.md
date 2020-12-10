# Eqsormo: Equilibrium sorting models in Python

Eqsormo implements equilibrium sorting models in Python. Equilibrium sorting models are models that simulate how households will re-sort across a metropolitan region in response to a change in policy or amenities. They allow evaluating non-marginal effects of policy changes---i.e. not only how the policies will affect the utility of people in the places they currently live, but how they will affect utility after households have re-sorted across the region in response to amenity and price changes.

It currently supports discrete-choice-based sorting models, specifically of the type implemented by Tra (2007, 2010, 2013), wherein the price enters the first stage of the model as a nonlinear budget term. This is in contrast to much of the literature where a constant price coefficient is estimated in the second stage. The advantage of the Tra approach is that by including price in the first stage, where there are also fixed effects for all housing types, there is less concern about unobserved housing attributes being correlated with price.

Eqsormo is currently alpha-stage software. Any results should be carefully evaluated, and issues reported to [the issue tracker](https://github.com/mattwigway/eqsormo/issues). API documentation [is available on readthedocs](https://eqsormo.readthedocs.io/en/latest/api.html#api); additional documentation is forthcoming.

Eqsormo can be installed with `pip`: `pip install eqsormo`

## References

Tra CI (2007) Evaluating the equilibrium welfare impacts of the 1990 Clean Air Act amendments in the Los Angeles area. Doctoral dissertation. University of Maryland, College Park, College Park, MD, USA. Available at: http://hdl.handle.net/1903/7236.

Tra CI (2010) A discrete choice equilibrium approach to valuing large environmental changes. Journal of Public Economics 94(1): 183–196. DOI: 10.1016/j.jpubeco.2009.10.006.

Tra CI (2013) Measuring the General Equilibrium Benefits of Air Quality Regulation in Small Urban Areas. Land Economics 89(2): 291–307.
