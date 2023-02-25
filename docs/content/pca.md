+++
title = "Principal component analysis"
menu = "main"
weight = 1
toc = true
aliases = ["pca"]
+++

## Resources

- [*Principal component analysis*](https://personal.utdallas.edu/~herve/abdi-awPCA2010.pdf) by Hervé Abdi and Lynne J. Williams is excellent at explaining PCA interpretation. It also covers some extensions to PCA.
- [*A Tutorial on Principal Component Analysis*](https://arxiv.org/pdf/1404.1100.pdf) by Jonathon Shlens goes into more detail on the intuition behind PCA, while also discussing its applicability and limits.

## Data

PCA assumes you have a dataframe consisting of numerical variables. This includes booleans and integers.

As an example, let's use a dataset describing the energy mix of each country in the world, in 2019. The dataset is normalized into percentages, which makes countries comparable with one another.


```python
import prince

dataset = prince.datasets.load_energy_mix(year=2019, normalize=True)
dataset.head().style.format('{:.0%}')  
```




<style type="text/css">
</style>
<table id="T_13108">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_13108_level0_col0" class="col_heading level0 col0" >coal</th>
      <th id="T_13108_level0_col1" class="col_heading level0 col1" >oil</th>
      <th id="T_13108_level0_col2" class="col_heading level0 col2" >gas</th>
      <th id="T_13108_level0_col3" class="col_heading level0 col3" >nuclear</th>
      <th id="T_13108_level0_col4" class="col_heading level0 col4" >hydro</th>
      <th id="T_13108_level0_col5" class="col_heading level0 col5" >wind</th>
      <th id="T_13108_level0_col6" class="col_heading level0 col6" >solar</th>
      <th id="T_13108_level0_col7" class="col_heading level0 col7" >other renewables</th>
    </tr>
    <tr>
      <th class="index_name level0" >continent</th>
      <th class="index_name level1" >country</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_13108_level0_row0" class="row_heading level0 row0" >Africa</th>
      <th id="T_13108_level1_row0" class="row_heading level1 row0" >Algeria</th>
      <td id="T_13108_row0_col0" class="data row0 col0" >1%</td>
      <td id="T_13108_row0_col1" class="data row0 col1" >35%</td>
      <td id="T_13108_row0_col2" class="data row0 col2" >64%</td>
      <td id="T_13108_row0_col3" class="data row0 col3" >0%</td>
      <td id="T_13108_row0_col4" class="data row0 col4" >0%</td>
      <td id="T_13108_row0_col5" class="data row0 col5" >0%</td>
      <td id="T_13108_row0_col6" class="data row0 col6" >0%</td>
      <td id="T_13108_row0_col7" class="data row0 col7" >0%</td>
    </tr>
    <tr>
      <th id="T_13108_level0_row1" class="row_heading level0 row1" >South America</th>
      <th id="T_13108_level1_row1" class="row_heading level1 row1" >Argentina</th>
      <td id="T_13108_row1_col0" class="data row1 col0" >1%</td>
      <td id="T_13108_row1_col1" class="data row1 col1" >35%</td>
      <td id="T_13108_row1_col2" class="data row1 col2" >50%</td>
      <td id="T_13108_row1_col3" class="data row1 col3" >2%</td>
      <td id="T_13108_row1_col4" class="data row1 col4" >10%</td>
      <td id="T_13108_row1_col5" class="data row1 col5" >1%</td>
      <td id="T_13108_row1_col6" class="data row1 col6" >0%</td>
      <td id="T_13108_row1_col7" class="data row1 col7" >1%</td>
    </tr>
    <tr>
      <th id="T_13108_level0_row2" class="row_heading level0 row2" >Oceania</th>
      <th id="T_13108_level1_row2" class="row_heading level1 row2" >Australia</th>
      <td id="T_13108_row2_col0" class="data row2 col0" >28%</td>
      <td id="T_13108_row2_col1" class="data row2 col1" >34%</td>
      <td id="T_13108_row2_col2" class="data row2 col2" >30%</td>
      <td id="T_13108_row2_col3" class="data row2 col3" >0%</td>
      <td id="T_13108_row2_col4" class="data row2 col4" >2%</td>
      <td id="T_13108_row2_col5" class="data row2 col5" >3%</td>
      <td id="T_13108_row2_col6" class="data row2 col6" >3%</td>
      <td id="T_13108_row2_col7" class="data row2 col7" >1%</td>
    </tr>
    <tr>
      <th id="T_13108_level0_row3" class="row_heading level0 row3" >Europe</th>
      <th id="T_13108_level1_row3" class="row_heading level1 row3" >Austria</th>
      <td id="T_13108_row3_col0" class="data row3 col0" >9%</td>
      <td id="T_13108_row3_col1" class="data row3 col1" >37%</td>
      <td id="T_13108_row3_col2" class="data row3 col2" >22%</td>
      <td id="T_13108_row3_col3" class="data row3 col3" >0%</td>
      <td id="T_13108_row3_col4" class="data row3 col4" >25%</td>
      <td id="T_13108_row3_col5" class="data row3 col5" >4%</td>
      <td id="T_13108_row3_col6" class="data row3 col6" >1%</td>
      <td id="T_13108_row3_col7" class="data row3 col7" >3%</td>
    </tr>
    <tr>
      <th id="T_13108_level0_row4" class="row_heading level0 row4" >Asia</th>
      <th id="T_13108_level1_row4" class="row_heading level1 row4" >Azerbaijan</th>
      <td id="T_13108_row4_col0" class="data row4 col0" >0%</td>
      <td id="T_13108_row4_col1" class="data row4 col1" >33%</td>
      <td id="T_13108_row4_col2" class="data row4 col2" >65%</td>
      <td id="T_13108_row4_col3" class="data row4 col3" >0%</td>
      <td id="T_13108_row4_col4" class="data row4 col4" >2%</td>
      <td id="T_13108_row4_col5" class="data row4 col5" >0%</td>
      <td id="T_13108_row4_col6" class="data row4 col6" >0%</td>
      <td id="T_13108_row4_col7" class="data row4 col7" >0%</td>
    </tr>
  </tbody>
</table>




## Fitting

The `PCA` estimator implements the `fit/transform` API from scikit-learn.


```python
pca = prince.PCA(
    n_components=4,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)
pca = pca.fit(dataset)
```

The available parameters are:

- `n_components` — the number of components that are computed. You only need two if your intention is to visualize the two major components.
- `n_iter` — the number of iterations used for computing the SVD.
- `rescale_with_mean` — whether to substract each column's mean
- `rescale_with_std` — whether to divide each column by it's standard deviation
- `copy` — if `False` then the computations will be done inplace which can have possible side-effects on the input data
- `engine` — what SVD engine to use (should be one of `['fbpca', 'sklearn']`)
- `random_state` — controls the randomness of the SVD results.

These methods are common to other Prince estimators.

## Eigenvalues

The importance of a principal component is indicated by the proportion of dataset inertia it explains. This is called the explained inertia, and is obtained by dividing the eigenvalues obtained with SVD by the total inertia.

The ideal situation is when a large share of inertia is explained by a low number of principal components.


```python
pca.eigenvalues_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eigenvalue</th>
      <th>% of variance</th>
      <th>% of variance (cumulative)</th>
    </tr>
    <tr>
      <th>component</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.963</td>
      <td>24.54%</td>
      <td>24.54%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.561</td>
      <td>19.51%</td>
      <td>44.05%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.403</td>
      <td>17.54%</td>
      <td>61.59%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.088</td>
      <td>13.60%</td>
      <td>75.19%</td>
    </tr>
  </tbody>
</table>
</div>



In this dataset, the first four components explain 75% of the dataset inertia. This isn't great, but isn't bad either. Eigenvalues can also be visualized with a [scree plot](https://www.wikiwand.com/en/Scree_plot).


```python
pca.scree_plot()
```





<div id="altair-viz-4e5993fb554b457b819f6ed7d653b796"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-4e5993fb554b457b819f6ed7d653b796") {
      outputDiv = document.getElementById("altair-viz-4e5993fb554b457b819f6ed7d653b796");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-6fcf8330645eee19b2933252fc53e2fc"}, "mark": {"type": "bar", "size": 10}, "encoding": {"tooltip": [{"field": "component", "type": "nominal"}, {"field": "eigenvalue", "type": "quantitative"}, {"field": "% of variance", "type": "quantitative"}, {"field": "% of variance (cumulative)", "type": "quantitative"}], "x": {"field": "component", "type": "nominal"}, "y": {"field": "eigenvalue", "type": "quantitative"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-6fcf8330645eee19b2933252fc53e2fc": [{"component": "0", "eigenvalue": 1.9630121441911277, "% of variance": 24.537651802389096, "% of variance (cumulative)": 24.537651802389096}, {"component": "1", "eigenvalue": 1.5606998579915559, "% of variance": 19.508748224894447, "% of variance (cumulative)": 44.04640002728354}, {"component": "2", "eigenvalue": 1.4032782280687068, "% of variance": 17.540977850858834, "% of variance (cumulative)": 61.58737787814238}, {"component": "3", "eigenvalue": 1.088220662211763, "% of variance": 13.602758277647037, "% of variance (cumulative)": 75.19013615578942}]}}, {"mode": "vega-lite"});
</script>



## Coordinates

### Rows

The row principal coordinates can be obtained once the `PCA` has been fitted to the data.


```python
pca.transform(dataset).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>component</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Africa</th>
      <th>Algeria</th>
      <td>-2.189068</td>
      <td>0.380243</td>
      <td>-0.388572</td>
      <td>0.336561</td>
    </tr>
    <tr>
      <th>South America</th>
      <th>Argentina</th>
      <td>-1.244981</td>
      <td>0.801917</td>
      <td>-0.389456</td>
      <td>0.293335</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <th>Australia</th>
      <td>0.203969</td>
      <td>-1.470254</td>
      <td>0.531819</td>
      <td>0.202603</td>
    </tr>
    <tr>
      <th>Europe</th>
      <th>Austria</th>
      <td>0.847122</td>
      <td>1.008296</td>
      <td>-0.521998</td>
      <td>-0.214031</td>
    </tr>
    <tr>
      <th>Asia</th>
      <th>Azerbaijan</th>
      <td>-2.190535</td>
      <td>0.632250</td>
      <td>-0.365515</td>
      <td>0.344389</td>
    </tr>
  </tbody>
</table>
</div>



The `transform` method is in fact an alias for `row_coordinates`:


```python
pca.transform(dataset).equals(pca.row_coordinates(dataset))
```




    True



This is transforming the original dataset into factor scores.

### Columns

The column coordinates are obtained during the PCA training process.


```python
pca.column_coordinates_
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>component</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>variable</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coal</th>
      <td>0.230205</td>
      <td>-0.395474</td>
      <td>0.740748</td>
      <td>-0.409552</td>
    </tr>
    <tr>
      <th>oil</th>
      <td>0.176200</td>
      <td>-0.418917</td>
      <td>-0.737855</td>
      <td>-0.330527</td>
    </tr>
    <tr>
      <th>gas</th>
      <td>-0.866927</td>
      <td>0.182990</td>
      <td>-0.096857</td>
      <td>0.369908</td>
    </tr>
    <tr>
      <th>nuclear</th>
      <td>0.310313</td>
      <td>-0.002598</td>
      <td>0.400608</td>
      <td>0.612367</td>
    </tr>
    <tr>
      <th>hydro</th>
      <td>0.440383</td>
      <td>0.744815</td>
      <td>-0.016375</td>
      <td>-0.179789</td>
    </tr>
    <tr>
      <th>wind</th>
      <td>0.518712</td>
      <td>-0.161507</td>
      <td>-0.364593</td>
      <td>0.413134</td>
    </tr>
    <tr>
      <th>solar</th>
      <td>0.415677</td>
      <td>-0.589288</td>
      <td>0.001012</td>
      <td>0.308925</td>
    </tr>
    <tr>
      <th>other renewables</th>
      <td>0.628750</td>
      <td>0.516935</td>
      <td>-0.084114</td>
      <td>0.031241</td>
    </tr>
  </tbody>
</table>
</div>



### Visualization

The row and column coordinates be visualized together with a scatter chart.


```python
pca.plot(
    dataset,
    x_component=0,
    y_component=1,
    color_by='continent',
    show_rows=True,
    show_columns=True
)
```





<div id="altair-viz-2d626afd04924e64be9fe5abcc8678a4"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-2d626afd04924e64be9fe5abcc8678a4") {
      outputDiv = document.getElementById("altair-viz-2d626afd04924e64be9fe5abcc8678a4");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": {"type": "circle", "size": 50}, "encoding": {"color": {"field": "continent", "type": "nominal"}, "tooltip": [{"field": "continent", "type": "nominal"}, {"field": "country", "type": "nominal"}, {"field": "component 0", "type": "quantitative"}, {"field": "component 1", "type": "quantitative"}], "x": {"axis": {"title": "component 0 \u2014 24.54%"}, "field": "component 0", "scale": {"zero": false}, "type": "quantitative"}, "y": {"axis": {"title": "component 1 \u2014 19.51%"}, "field": "component 1", "scale": {"zero": false}, "type": "quantitative"}}}, {"data": {"name": "data-725ffad0ca4c6d9b193a9a8f6761f655"}, "mark": {"type": "square", "color": "green", "size": 50}, "encoding": {"tooltip": [{"field": "variable", "type": "nominal"}], "x": {"field": "component 0", "scale": {"zero": false}, "type": "quantitative"}, "y": {"field": "component 1", "scale": {"zero": false}, "type": "quantitative"}}}], "data": {"name": "data-355b83fdc63421421c1ce95fd2806f7d"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-355b83fdc63421421c1ce95fd2806f7d": [{"continent": "Africa", "country": "Algeria", "component 0": -2.1890683235018233, "component 1": 0.3802430845833857, "component 2": -0.38857200885287535, "component 3": 0.3365614451539336}, {"continent": "South America", "country": "Argentina", "component 0": -1.2449809136844547, "component 1": 0.8019170914196425, "component 2": -0.3894555350426134, "component 3": 0.29333540618729065}, {"continent": "Oceania", "country": "Australia", "component 0": 0.20396903697829905, "component 1": -1.470253963268879, "component 2": 0.5318189014543244, "component 3": 0.20260263484070937}, {"continent": "Europe", "country": "Austria", "component 0": 0.8471216351969489, "component 1": 1.0082962135958828, "component 2": -0.5219975517003386, "component 3": -0.21403111456821397}, {"continent": "Asia", "country": "Azerbaijan", "component 0": -2.190534789276698, "component 1": 0.6322502201648152, "component 2": -0.3655152611379308, "component 3": 0.3443892197996473}, {"continent": "Asia", "country": "Bangladesh", "component 0": -2.3971270086304, "component 1": 0.5978805199941836, "component 2": 0.40469847963659195, "component 3": 0.5386494186468507}, {"continent": "Europe", "country": "Belarus", "component 0": -2.249012696615155, "component 1": 0.5130378176751523, "component 2": -0.11468609899301206, "component 3": 0.3560478317686008}, {"continent": "Europe", "country": "Belgium", "component 0": 0.5129552509797374, "component 1": -0.7918525399485158, "component 2": -0.400805133561515, "component 3": 1.0953031988628117}, {"continent": "South America", "country": "Brazil", "component 0": 1.4326021939469056, "component 1": 1.6174697914843033, "component 2": -0.7258786903436064, "component 3": -0.5299045666708682}, {"continent": "Europe", "country": "Bulgaria", "component 0": 1.1854225672961205, "component 1": -1.0657556131268628, "component 2": 1.7702403719829118, "component 3": 1.3180849896906877}, {"continent": "North America", "country": "Canada", "component 0": -0.04935611938254242, "component 1": 1.3314516816840904, "component 2": 0.0184727703938132, "component 3": 0.18213521686695844}, {"continent": "South America", "country": "Chile", "component 0": 1.870148862132147, "component 1": -1.0171233793722603, "component 2": -0.4074232445136813, "component 3": 0.14923123329021906}, {"continent": "Asia", "country": "China", "component 0": 0.8976977193613257, "component 1": -0.9862977117676487, "component 2": 2.363994882035523, "component 3": -0.8765855446859865}, {"continent": "South America", "country": "Colombia", "component 0": -0.21546953617959347, "component 1": 1.235349594629521, "component 2": 0.10620523150315202, "component 3": -1.0219289784382288}, {"continent": "Europe", "country": "Croatia", "component 0": -0.026051450970706772, "component 1": 0.8354359363062168, "component 2": -0.774255397318505, "component 3": -0.2241760596626515}, {"continent": "Asia", "country": "Cyprus", "component 0": 0.8687868577482043, "component 1": -2.089658829543803, "component 2": -2.695608316196079, "component 3": -1.2490327028693906}, {"continent": "Europe", "country": "Czechia", "component 0": 0.6274389636722499, "component 1": -0.5459233709325043, "component 2": 2.071009507937302, "component 3": 0.5558323999885885}, {"continent": "Europe", "country": "Denmark", "component 0": 3.0954896363993383, "component 1": -0.6040755184733296, "component 2": -2.602856386619385, "component 3": 1.9526269634805182}, {"continent": "South America", "country": "Ecuador", "component 0": 0.6808765946846796, "component 1": 0.8603121903708817, "component 2": -1.4821750668448872, "component 3": -1.7058733927033385}, {"continent": "Africa", "country": "Egypt", "component 0": -1.5166340436710188, "component 1": 0.009928284038989089, "component 2": -0.5178544939343204, "component 3": 0.30166122566434267}, {"continent": "Europe", "country": "Estonia", "component 0": 0.8785647112733624, "component 1": -0.5073678406599711, "component 2": 1.972950295963322, "component 3": -1.4709428834945577}, {"continent": "Europe", "country": "Finland", "component 0": 2.290822972733853, "component 1": 1.3361049807594134, "component 2": 0.4412819223445058, "component 3": 1.0244709859122505}, {"continent": "Europe", "country": "France", "component 0": 1.176724338916868, "component 1": -0.15904669014179965, "component 2": 1.3000451899880228, "component 3": 3.0163314636307432}, {"continent": "Europe", "country": "Germany", "component 0": 1.6987226706793859, "component 1": -1.6244462947865197, "component 2": -0.27013965709612114, "component 3": 1.6493006638739165}, {"continent": "Europe", "country": "Greece", "component 0": 1.1641812187144118, "component 1": -2.19599177029954, "component 2": -1.261484748709455, "component 3": 0.3354366669156622}, {"continent": "Asia", "country": "Hong Kong", "component 0": -0.18207387572526562, "component 1": -0.9960252506516332, "component 2": -0.8216993717345471, "component 3": -1.8123368646615845}, {"continent": "Europe", "country": "Hungary", "component 0": -0.17172022935545586, "component 1": -0.3171973578019041, "component 2": 0.4412107040751169, "component 3": 1.2202072718048358}, {"continent": "Europe", "country": "Iceland", "component 0": 4.261623746019558, "component 1": 6.181965785949938, "component 2": -0.0494473774179157, "component 3": -1.0734253159087248}, {"continent": "Asia", "country": "India", "component 0": 0.7067920410516929, "component 1": -1.2084451380419892, "component 2": 1.8921935429102172, "component 3": -1.2290097255023582}, {"continent": "Asia", "country": "Indonesia", "component 0": -0.253397015857427, "component 1": -0.35062638725575496, "component 2": 1.0012779252240034, "component 3": -1.480586786707095}, {"continent": "Asia", "country": "Iran", "component 0": -2.2430866645115692, "component 1": 0.6641418816126279, "component 2": -0.2753243349531549, "component 3": 0.36550644693351847}, {"continent": "Asia", "country": "Iraq", "component 0": -1.0580052057598144, "component 1": -0.3503945146153497, "component 2": -1.5499735586527297, "component 3": -0.8877620575006854}, {"continent": "Europe", "country": "Ireland", "component 0": 0.5336963001246187, "component 1": -0.4485682100956032, "component 2": -1.8190945994870873, "component 3": 0.7914914028828495}, {"continent": "Asia", "country": "Israel", "component 0": -0.4335780208045791, "component 1": -1.3590787033978484, "component 2": -0.00982205788815815, "component 3": -0.13037765027458015}, {"continent": "Europe", "country": "Italy", "component 0": 0.5310378262342746, "component 1": -0.9881351985051243, "component 2": -0.6555419826155741, "component 3": 1.0601878593436749}, {"continent": "Asia", "country": "Japan", "component 0": 0.9244620897722294, "component 1": -1.8774672876612908, "component 2": 0.562496372765546, "component 3": 0.26949020366384685}, {"continent": "Asia", "country": "Kazakhstan", "component 0": -0.4712425702426596, "component 1": -0.4409135299104643, "component 2": 2.2147867824958647, "component 3": -1.432305550294615}, {"continent": "Asia", "country": "Kuwait", "component 0": -1.7748853192929983, "component 1": 0.09643076396919852, "component 2": -0.8635229264379645, "component 3": -0.16405395685138974}, {"continent": "Europe", "country": "Latvia", "component 0": 0.017284536269702066, "component 1": 1.1793376092784738, "component 2": -1.011450597901127, "component 3": -0.644764818567091}, {"continent": "Europe", "country": "Lithuania", "component 0": -0.15035909811193407, "component 1": -0.28967543229430537, "component 2": -1.516006048025831, "component 3": -0.049031648861428634}, {"continent": "Europe", "country": "Luxembourg", "component 0": 0.020479631335571843, "component 1": -0.9037156673027422, "component 2": -2.001629098524809, "component 3": -0.9818322149959696}, {"continent": "Asia", "country": "Malaysia", "component 0": -0.9976183920795465, "component 1": 0.09023205134833762, "component 2": 0.37269472544858684, "component 3": -0.7445698325739817}, {"continent": "North America", "country": "Mexico", "component 0": -0.592788752192353, "component 1": -0.5044827811882522, "component 2": -0.5464486342203411, "component 3": 0.38230851641816466}, {"continent": "Africa", "country": "Morocco", "component 0": 0.9505858665860268, "component 1": -1.8601647853914747, "component 2": -0.47433965213458223, "component 3": -0.9268221096966671}, {"continent": "Europe", "country": "Netherlands", "component 0": -0.3677161721064315, "component 1": -0.700778674056845, "component 2": -0.7817802429951419, "component 3": 0.28025768695664566}, {"continent": "Oceania", "country": "New Zealand", "component 0": 1.1874655793567415, "component 1": 2.106798494577382, "component 2": -0.56114920151885, "component 3": -0.7012173845436392}, {"continent": "Europe", "country": "North Macedonia", "component 0": 0.20630503558060567, "component 1": -0.34293744791685327, "component 2": 0.926578726742162, "component 3": -1.6390023774000688}, {"continent": "Europe", "country": "Norway", "component 0": 1.2610276286722495, "component 1": 3.434346314734909, "component 2": 0.02985717657952377, "component 3": -0.9979450148949482}, {"continent": "Asia", "country": "Oman", "component 0": -2.0859852511976067, "component 1": 0.3506924216317761, "component 2": -0.5926625994332704, "component 3": 0.09221727106200059}, {"continent": "Asia", "country": "Pakistan", "component 0": -1.1124312900847926, "component 1": 0.5706273023800105, "component 2": 0.5735798414309855, "component 3": 0.10570158286083}, {"continent": "South America", "country": "Peru", "component 0": 0.076343965897632, "component 1": 1.0433687069166357, "component 2": -0.7441762552237378, "component 3": -0.5838909236488903}, {"continent": "Asia", "country": "Philippines", "component 0": 0.8894342072409059, "component 1": -0.24689641490558673, "component 2": 0.5587667860124724, "component 3": -1.4789896452705176}, {"continent": "Europe", "country": "Poland", "component 0": 0.11371335920010885, "component 1": -0.5857586276139334, "component 2": 1.2282640075958857, "component 3": -1.0636376237589085}, {"continent": "Europe", "country": "Portugal", "component 0": 1.3844655532695138, "component 1": -0.4524740714133119, "component 2": -1.7529969439179967, "component 3": 0.7913714592379008}, {"continent": "Asia", "country": "Qatar", "component 0": -2.5816881977308666, "component 1": 0.7363708049471228, "component 2": -0.145774207031997, "component 3": 0.5925799250304673}, {"continent": "Europe", "country": "Romania", "component 0": 0.25827830907700455, "component 1": -0.1778903725331528, "component 2": 0.20806232182943288, "component 3": 0.7112128576047942}, {"continent": "Europe", "country": "Russia", "component 0": -1.5975710877684557, "component 1": 0.7482393765483546, "component 2": 0.8398588569996351, "component 3": 0.495304508249283}, {"continent": "Asia", "country": "Saudi Arabia", "component 0": -1.2169511229281027, "component 1": -0.34523765455730027, "component 2": -1.4084090472883335, "component 3": -0.6743682662778391}, {"continent": "Asia", "country": "Singapore", "component 0": -0.33198779523005784, "component 1": -0.9423859681500143, "component 2": -2.2075233740086415, "component 3": -1.584638147184922}, {"continent": "Europe", "country": "Slovakia", "component 0": 0.19621358496738664, "component 1": 0.19712575555516065, "component 2": 1.6149297395661737, "component 3": 1.2754898966328727}, {"continent": "Europe", "country": "Slovenia", "component 0": 0.7777098608044026, "component 1": 0.14121326856371014, "component 2": 1.018454759984398, "component 3": 0.5192727429857225}, {"continent": "Africa", "country": "South Africa", "component 0": 0.5767047177260597, "component 1": -1.4441788820576515, "component 2": 2.92878968940891, "component 3": -1.519839744591912}, {"continent": "Asia", "country": "South Korea", "component 0": 0.21255919897857214, "component 1": -0.8402899582816511, "component 2": 0.9040241202682368, "component 3": -0.1822739647532943}, {"continent": "Europe", "country": "Spain", "component 0": 1.2887741609981125, "component 1": -1.3591702196892823, "component 2": -1.0402238383885598, "component 3": 1.6998483460311646}, {"continent": "Asia", "country": "Sri Lanka", "component 0": 0.7667849113239487, "component 1": -0.849815150314963, "component 2": -0.9378616716015267, "component 3": -1.6243052175945731}, {"continent": "Europe", "country": "Sweden", "component 0": 2.701213347286986, "component 1": 1.7632391683044362, "component 2": 0.6496430133001325, "component 3": 2.1287716784050734}, {"continent": "Europe", "country": "Switzerland", "component 0": 1.362772095537243, "component 1": 0.6501185516707244, "component 2": 0.36092090701766094, "component 3": 0.9284928426310205}, {"continent": "Asia", "country": "Taiwan", "component 0": -0.004197824507348786, "component 1": -0.8782300182476133, "component 2": 1.0279478546885539, "component 3": -0.640344094375609}, {"continent": "Asia", "country": "Thailand", "component 0": -0.4932957709392967, "component 1": -0.36354288186353073, "component 2": -0.48103035604310834, "component 3": -0.5015628496667942}, {"continent": "North America", "country": "Trinidad and Tobago", "component 0": -3.154195968923132, "component 1": 1.1571089913273234, "component 2": 0.3960492989914883, "component 3": 1.152986646261689}, {"continent": "Asia", "country": "Turkey", "component 0": 0.48907944384074975, "component 1": -0.24278543765700988, "component 2": 0.5178495315387395, "component 3": -0.25221446991635743}, {"continent": "Asia", "country": "Turkmenistan", "component 0": -2.7846323628628515, "component 1": 0.877772686226518, "component 2": 0.04176680205759131, "component 3": 0.7860878218398635}, {"continent": "Europe", "country": "Ukraine", "component 0": -0.1381753195659796, "component 1": -0.19800889876224256, "component 2": 2.6915173157901426, "component 3": 1.3700987939825278}, {"continent": "Asia", "country": "United Arab Emirates", "component 0": -1.7351268299856146, "component 1": -0.10457038663938835, "component 2": -0.5342744711637439, "component 3": 0.24761198276763993}, {"continent": "Europe", "country": "United Kingdom", "component 0": 0.6423607687028626, "component 1": -0.32629836707261894, "component 2": -0.8481142361091378, "component 3": 1.489797274100862}, {"continent": "North America", "country": "United States", "component 0": -0.1444837390548087, "component 1": -0.4650541756038503, "component 2": 0.04667262341160476, "component 3": 0.6557617419450326}, {"continent": "Asia", "country": "Uzbekistan", "component 0": -3.0562470286777024, "component 1": 1.3756770657885564, "component 2": 0.7986303616593644, "component 3": 1.124361025845009}, {"continent": "South America", "country": "Venezuela", "component 0": -0.9960196887867527, "component 1": 1.6832972113795317, "component 2": -0.24758555952975558, "component 3": -0.36997570687202563}, {"continent": "Asia", "country": "Vietnam", "component 0": 0.43900247962720396, "component 1": -0.38879424564503096, "component 2": 1.9690284940800515, "component 3": -1.524859543811323}], "data-725ffad0ca4c6d9b193a9a8f6761f655": [{"variable": "coal", "component 0": 0.230205059218531, "component 1": -0.3954739928509997, "component 2": 0.7407478667651939, "component 3": -0.40955234702204707}, {"variable": "oil", "component 0": 0.17619968065162395, "component 1": -0.4189171848047792, "component 2": -0.7378549806407537, "component 3": -0.33052676404393033}, {"variable": "gas", "component 0": -0.8669272766513348, "component 1": 0.18298963620780567, "component 2": -0.09685698011110348, "component 3": 0.3699081264138578}, {"variable": "nuclear", "component 0": 0.3103131366782066, "component 1": -0.0025975015498222987, "component 2": 0.4006084663681456, "component 3": 0.6123671369080909}, {"variable": "hydro", "component 0": 0.4403831610176404, "component 1": 0.7448154202060033, "component 2": -0.0163747349746629, "component 3": -0.179789011947019}, {"variable": "wind", "component 0": 0.5187123415283995, "component 1": -0.1615073635769731, "component 2": -0.36459313994754994, "component 3": 0.41313382022413225}, {"variable": "solar", "component 0": 0.4156773780406406, "component 1": -0.5892876119685019, "component 2": 0.0010116756680984193, "component 3": 0.3089245697368804}, {"variable": "other renewables", "component 0": 0.6287501866925463, "component 1": 0.5169352851041867, "component 2": -0.08411374705190805, "component 3": 0.03124149711735093}]}}, {"mode": "vega-lite"});
</script>



## Contributions


```python
pca.row_contributions_.head().style.format('{:.0%}')  
```




<style type="text/css">
</style>
<table id="T_46b75">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="index_name level0" >component</th>
      <th id="T_46b75_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_46b75_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_46b75_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_46b75_level0_col3" class="col_heading level0 col3" >3</th>
    </tr>
    <tr>
      <th class="index_name level0" >continent</th>
      <th class="index_name level1" >country</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_46b75_level0_row0" class="row_heading level0 row0" >Africa</th>
      <th id="T_46b75_level1_row0" class="row_heading level1 row0" >Algeria</th>
      <td id="T_46b75_row0_col0" class="data row0 col0" >3%</td>
      <td id="T_46b75_row0_col1" class="data row0 col1" >0%</td>
      <td id="T_46b75_row0_col2" class="data row0 col2" >0%</td>
      <td id="T_46b75_row0_col3" class="data row0 col3" >0%</td>
    </tr>
    <tr>
      <th id="T_46b75_level0_row1" class="row_heading level0 row1" >South America</th>
      <th id="T_46b75_level1_row1" class="row_heading level1 row1" >Argentina</th>
      <td id="T_46b75_row1_col0" class="data row1 col0" >1%</td>
      <td id="T_46b75_row1_col1" class="data row1 col1" >1%</td>
      <td id="T_46b75_row1_col2" class="data row1 col2" >0%</td>
      <td id="T_46b75_row1_col3" class="data row1 col3" >0%</td>
    </tr>
    <tr>
      <th id="T_46b75_level0_row2" class="row_heading level0 row2" >Oceania</th>
      <th id="T_46b75_level1_row2" class="row_heading level1 row2" >Australia</th>
      <td id="T_46b75_row2_col0" class="data row2 col0" >0%</td>
      <td id="T_46b75_row2_col1" class="data row2 col1" >2%</td>
      <td id="T_46b75_row2_col2" class="data row2 col2" >0%</td>
      <td id="T_46b75_row2_col3" class="data row2 col3" >0%</td>
    </tr>
    <tr>
      <th id="T_46b75_level0_row3" class="row_heading level0 row3" >Europe</th>
      <th id="T_46b75_level1_row3" class="row_heading level1 row3" >Austria</th>
      <td id="T_46b75_row3_col0" class="data row3 col0" >0%</td>
      <td id="T_46b75_row3_col1" class="data row3 col1" >1%</td>
      <td id="T_46b75_row3_col2" class="data row3 col2" >0%</td>
      <td id="T_46b75_row3_col3" class="data row3 col3" >0%</td>
    </tr>
    <tr>
      <th id="T_46b75_level0_row4" class="row_heading level0 row4" >Asia</th>
      <th id="T_46b75_level1_row4" class="row_heading level1 row4" >Azerbaijan</th>
      <td id="T_46b75_row4_col0" class="data row4 col0" >3%</td>
      <td id="T_46b75_row4_col1" class="data row4 col1" >0%</td>
      <td id="T_46b75_row4_col2" class="data row4 col2" >0%</td>
      <td id="T_46b75_row4_col3" class="data row4 col3" >0%</td>
    </tr>
  </tbody>
</table>




Observations with high contributions and different signs can be opposed to help interpret the component, because these observations represent the two endpoints of this component.

Column contributions are also available.


```python
pca.column_contributions_.style.format('{:.0%}')  
```




<style type="text/css">
</style>
<table id="T_01c2b">
  <thead>
    <tr>
      <th class="index_name level0" >component</th>
      <th id="T_01c2b_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_01c2b_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_01c2b_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_01c2b_level0_col3" class="col_heading level0 col3" >3</th>
    </tr>
    <tr>
      <th class="index_name level0" >variable</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_01c2b_level0_row0" class="row_heading level0 row0" >coal</th>
      <td id="T_01c2b_row0_col0" class="data row0 col0" >3%</td>
      <td id="T_01c2b_row0_col1" class="data row0 col1" >10%</td>
      <td id="T_01c2b_row0_col2" class="data row0 col2" >39%</td>
      <td id="T_01c2b_row0_col3" class="data row0 col3" >15%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row1" class="row_heading level0 row1" >oil</th>
      <td id="T_01c2b_row1_col0" class="data row1 col0" >2%</td>
      <td id="T_01c2b_row1_col1" class="data row1 col1" >11%</td>
      <td id="T_01c2b_row1_col2" class="data row1 col2" >39%</td>
      <td id="T_01c2b_row1_col3" class="data row1 col3" >10%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row2" class="row_heading level0 row2" >gas</th>
      <td id="T_01c2b_row2_col0" class="data row2 col0" >38%</td>
      <td id="T_01c2b_row2_col1" class="data row2 col1" >2%</td>
      <td id="T_01c2b_row2_col2" class="data row2 col2" >1%</td>
      <td id="T_01c2b_row2_col3" class="data row2 col3" >13%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row3" class="row_heading level0 row3" >nuclear</th>
      <td id="T_01c2b_row3_col0" class="data row3 col0" >5%</td>
      <td id="T_01c2b_row3_col1" class="data row3 col1" >0%</td>
      <td id="T_01c2b_row3_col2" class="data row3 col2" >11%</td>
      <td id="T_01c2b_row3_col3" class="data row3 col3" >34%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row4" class="row_heading level0 row4" >hydro</th>
      <td id="T_01c2b_row4_col0" class="data row4 col0" >10%</td>
      <td id="T_01c2b_row4_col1" class="data row4 col1" >36%</td>
      <td id="T_01c2b_row4_col2" class="data row4 col2" >0%</td>
      <td id="T_01c2b_row4_col3" class="data row4 col3" >3%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row5" class="row_heading level0 row5" >wind</th>
      <td id="T_01c2b_row5_col0" class="data row5 col0" >14%</td>
      <td id="T_01c2b_row5_col1" class="data row5 col1" >2%</td>
      <td id="T_01c2b_row5_col2" class="data row5 col2" >9%</td>
      <td id="T_01c2b_row5_col3" class="data row5 col3" >16%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row6" class="row_heading level0 row6" >solar</th>
      <td id="T_01c2b_row6_col0" class="data row6 col0" >9%</td>
      <td id="T_01c2b_row6_col1" class="data row6 col1" >22%</td>
      <td id="T_01c2b_row6_col2" class="data row6 col2" >0%</td>
      <td id="T_01c2b_row6_col3" class="data row6 col3" >9%</td>
    </tr>
    <tr>
      <th id="T_01c2b_level0_row7" class="row_heading level0 row7" >other renewables</th>
      <td id="T_01c2b_row7_col0" class="data row7 col0" >20%</td>
      <td id="T_01c2b_row7_col1" class="data row7 col1" >17%</td>
      <td id="T_01c2b_row7_col2" class="data row7 col2" >1%</td>
      <td id="T_01c2b_row7_col3" class="data row7 col3" >0%</td>
    </tr>
  </tbody>
</table>




## Cosine similarities


```python
pca.row_cosine_similarities(dataset).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>component</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Africa</th>
      <th>Algeria</th>
      <td>0.902764</td>
      <td>0.027238</td>
      <td>0.028445</td>
      <td>0.021340</td>
    </tr>
    <tr>
      <th>South America</th>
      <th>Argentina</th>
      <td>0.626517</td>
      <td>0.259936</td>
      <td>0.061309</td>
      <td>0.034781</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <th>Australia</th>
      <td>0.008468</td>
      <td>0.439999</td>
      <td>0.057570</td>
      <td>0.008355</td>
    </tr>
    <tr>
      <th>Europe</th>
      <th>Austria</th>
      <td>0.236087</td>
      <td>0.334470</td>
      <td>0.089643</td>
      <td>0.015071</td>
    </tr>
    <tr>
      <th>Asia</th>
      <th>Azerbaijan</th>
      <td>0.868801</td>
      <td>0.072377</td>
      <td>0.024190</td>
      <td>0.021474</td>
    </tr>
  </tbody>
</table>
</div>




```python
pca.column_cosine_similarities_
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>component</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>variable</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coal</th>
      <td>0.052994</td>
      <td>0.156400</td>
      <td>0.548707</td>
      <td>0.167733</td>
    </tr>
    <tr>
      <th>oil</th>
      <td>0.031046</td>
      <td>0.175492</td>
      <td>0.544430</td>
      <td>0.109248</td>
    </tr>
    <tr>
      <th>gas</th>
      <td>0.751563</td>
      <td>0.033485</td>
      <td>0.009381</td>
      <td>0.136832</td>
    </tr>
    <tr>
      <th>nuclear</th>
      <td>0.096294</td>
      <td>0.000007</td>
      <td>0.160487</td>
      <td>0.374994</td>
    </tr>
    <tr>
      <th>hydro</th>
      <td>0.193937</td>
      <td>0.554750</td>
      <td>0.000268</td>
      <td>0.032324</td>
    </tr>
    <tr>
      <th>wind</th>
      <td>0.269062</td>
      <td>0.026085</td>
      <td>0.132928</td>
      <td>0.170680</td>
    </tr>
    <tr>
      <th>solar</th>
      <td>0.172788</td>
      <td>0.347260</td>
      <td>0.000001</td>
      <td>0.095434</td>
    </tr>
    <tr>
      <th>other renewables</th>
      <td>0.395327</td>
      <td>0.267222</td>
      <td>0.007075</td>
      <td>0.000976</td>
    </tr>
  </tbody>
</table>
</div>



## Column correlations


```python
pca.column_correlations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>component</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>variable</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coal</th>
      <td>0.230205</td>
      <td>-0.395474</td>
      <td>0.740748</td>
      <td>-0.409552</td>
    </tr>
    <tr>
      <th>oil</th>
      <td>0.176200</td>
      <td>-0.418917</td>
      <td>-0.737855</td>
      <td>-0.330527</td>
    </tr>
    <tr>
      <th>gas</th>
      <td>-0.866927</td>
      <td>0.182990</td>
      <td>-0.096857</td>
      <td>0.369908</td>
    </tr>
    <tr>
      <th>nuclear</th>
      <td>0.310313</td>
      <td>-0.002598</td>
      <td>0.400608</td>
      <td>0.612367</td>
    </tr>
    <tr>
      <th>hydro</th>
      <td>0.440383</td>
      <td>0.744815</td>
      <td>-0.016375</td>
      <td>-0.179789</td>
    </tr>
    <tr>
      <th>wind</th>
      <td>0.518712</td>
      <td>-0.161507</td>
      <td>-0.364593</td>
      <td>0.413134</td>
    </tr>
    <tr>
      <th>solar</th>
      <td>0.415677</td>
      <td>-0.589288</td>
      <td>0.001012</td>
      <td>0.308925</td>
    </tr>
    <tr>
      <th>other renewables</th>
      <td>0.628750</td>
      <td>0.516935</td>
      <td>-0.084114</td>
      <td>0.031241</td>
    </tr>
  </tbody>
</table>
</div>




```python
(pca.column_correlations ** 2).equals(pca.column_cosine_similarities_)
```




    True



## Inverse transformation

You can transform row projections back into their original space by using the `inverse_transform` method.


```python
reconstructed = pca.inverse_transform(pca.transform(dataset))
reconstructed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Africa</th>
      <th>Algeria</th>
      <td>0.001834</td>
      <td>0.352874</td>
      <td>0.624762</td>
      <td>0.007815</td>
      <td>0.017921</td>
      <td>0.001360</td>
      <td>0.001560</td>
      <td>-0.008127</td>
    </tr>
    <tr>
      <th>South America</th>
      <th>Argentina</th>
      <td>0.008276</td>
      <td>0.351532</td>
      <td>0.509802</td>
      <td>0.021580</td>
      <td>0.084064</td>
      <td>0.010980</td>
      <td>0.002184</td>
      <td>0.011583</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <th>Australia</th>
      <td>0.273906</td>
      <td>0.415119</td>
      <td>0.223955</td>
      <td>0.065609</td>
      <td>-0.022641</td>
      <td>0.028642</td>
      <td>0.015652</td>
      <td>-0.000241</td>
    </tr>
    <tr>
      <th>Europe</th>
      <th>Austria</th>
      <td>0.075100</td>
      <td>0.421199</td>
      <td>0.203325</td>
      <td>0.030648</td>
      <td>0.187622</td>
      <td>0.031305</td>
      <td>0.005588</td>
      <td>0.045211</td>
    </tr>
    <tr>
      <th>Asia</th>
      <th>Azerbaijan</th>
      <td>-0.009791</td>
      <td>0.336696</td>
      <td>0.633039</td>
      <td>0.008686</td>
      <td>0.035554</td>
      <td>0.000077</td>
      <td>0.000492</td>
      <td>-0.004752</td>
    </tr>
  </tbody>
</table>
</div>



This allows measuring the reconstruction error of the PCA. This is usually defined as the $L^2$ norm between the reconstructed dataset and the original dataset.


```python
import numpy as np

np.linalg.norm(reconstructed.values - dataset.values, ord=2)
```




    0.8759916059569246



## Supplementary data

Active rows and columns make up the dataset you fit the PCA with. Anything you provide afterwards is considered supplementary data.

For example, we can fit the PCA on all countries that are not part of North America.


```python
active = dataset.query("continent != 'North America'")
pca = prince.PCA().fit(active)
```

The data for North America can still be projected onto the principal components.


```python
pca.transform(dataset).query("continent == 'North America'")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>component</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">North America</th>
      <th>Canada</th>
      <td>0.009864</td>
      <td>1.332859</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>-0.695023</td>
      <td>-0.402230</td>
    </tr>
    <tr>
      <th>Trinidad and Tobago</th>
      <td>-3.072833</td>
      <td>1.393067</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>-0.226122</td>
      <td>-0.433260</td>
    </tr>
  </tbody>
</table>
</div>



As for supplementary, they must be provided during the `fit` call.


```python
pca = prince.PCA().fit(dataset, supplementary_columns=['hydro', 'wind', 'solar'])
pca.column_correlations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>component</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>variable</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coal</th>
      <td>0.621633</td>
      <td>-0.380008</td>
    </tr>
    <tr>
      <th>oil</th>
      <td>0.048966</td>
      <td>0.948650</td>
    </tr>
    <tr>
      <th>gas</th>
      <td>-0.916078</td>
      <td>-0.331546</td>
    </tr>
    <tr>
      <th>nuclear</th>
      <td>0.384458</td>
      <td>-0.401675</td>
    </tr>
    <tr>
      <th>other renewables</th>
      <td>0.467116</td>
      <td>0.086656</td>
    </tr>
    <tr>
      <th>hydro</th>
      <td>0.246646</td>
      <td>0.037118</td>
    </tr>
    <tr>
      <th>wind</th>
      <td>0.195675</td>
      <td>0.184247</td>
    </tr>
    <tr>
      <th>solar</th>
      <td>0.251247</td>
      <td>0.076237</td>
    </tr>
  </tbody>
</table>
</div>



There can be supplementary rows and columns at the same time.


```python
pca = prince.PCA().fit(active, supplementary_columns=['hydro', 'wind', 'solar'])
```

## Performance

Under the hood, Prince uses a [randomised version of SVD](https://research.facebook.com/blog/2014/9/fast-randomized-svd/). This is much faster than traditional SVD. However, the results may have a small inherent randomness. This can be controlled with the `random_state` parameter. The accurary of the results can be increased by providing a higher `n_iter` parameter.

By default `prince` uses scikit-learn's randomized SVD implementation -- the one used in [`TruncatedSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

Prince supports different SVD backends. For the while, the only other supported randomized backend is [Facebook's randomized SVD implementation](https://research.facebook.com/blog/fast-randomized-svd/), called [fbpca](http://fbpca.readthedocs.org/en/latest/):


```python
prince.PCA(engine='fbpca')
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(engine=&#x27;fbpca&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(engine=&#x27;fbpca&#x27;)</pre></div></div></div></div></div>



You can also use a non-randomized SVD implementation, using the [`scipy.linalg.svd` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html):


```python
prince.PCA(engine='scipy')
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(engine=&#x27;scipy&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(engine=&#x27;scipy&#x27;)</pre></div></div></div></div></div>



Here is a very crude benchmark that compares each engine on different dataset sizes.


```python
import itertools
import numpy as np

N = [100, 10_000, 100_000]
P = [3, 10, 30]
ENGINES = ['sklearn', 'fbpca', 'scipy']

perf = []

for n, p, engine in itertools.product(N, P, ENGINES):

    # Too slow
    if engine == 'scipy' and n > 10_000:
        continue

    X = pd.DataFrame(np.random.random(size=(n, p)))
    duration = %timeit -q -n 1 -r 3 -o prince.PCA(engine=engine, n_iter=3).fit(X)
    perf.append({
        'n': n,
        'p': p,
        'engine': engine,
        'seconds': duration.average
    })
```


```python
(
    pd.DataFrame(perf)
    .set_index(['n', 'p'])
    .groupby(['n', 'p'], group_keys=False)
    .apply(lambda x: x[['engine', 'seconds']]
    .sort_values('duration'))
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>engine</th>
      <th>duration</th>
    </tr>
    <tr>
      <th>n</th>
      <th>p</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">100</th>
      <th>3</th>
      <td>fbpca</td>
      <td>0.000818</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sklearn</td>
      <td>0.003405</td>
    </tr>
    <tr>
      <th>3</th>
      <td>scipy</td>
      <td>0.022696</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fbpca</td>
      <td>0.002599</td>
    </tr>
    <tr>
      <th>10</th>
      <td>scipy</td>
      <td>0.003026</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sklearn</td>
      <td>0.006835</td>
    </tr>
    <tr>
      <th>30</th>
      <td>scipy</td>
      <td>0.001455</td>
    </tr>
    <tr>
      <th>30</th>
      <td>fbpca</td>
      <td>0.001526</td>
    </tr>
    <tr>
      <th>30</th>
      <td>sklearn</td>
      <td>0.004369</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">10000</th>
      <th>3</th>
      <td>fbpca</td>
      <td>0.003056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sklearn</td>
      <td>0.166077</td>
    </tr>
    <tr>
      <th>3</th>
      <td>scipy</td>
      <td>0.335397</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fbpca</td>
      <td>0.269857</td>
    </tr>
    <tr>
      <th>10</th>
      <td>scipy</td>
      <td>0.614554</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sklearn</td>
      <td>0.855684</td>
    </tr>
    <tr>
      <th>30</th>
      <td>sklearn</td>
      <td>1.067292</td>
    </tr>
    <tr>
      <th>30</th>
      <td>fbpca</td>
      <td>1.169046</td>
    </tr>
    <tr>
      <th>30</th>
      <td>scipy</td>
      <td>1.650976</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">100000</th>
      <th>3</th>
      <td>fbpca</td>
      <td>0.121617</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sklearn</td>
      <td>0.500503</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fbpca</td>
      <td>0.574221</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sklearn</td>
      <td>1.637214</td>
    </tr>
    <tr>
      <th>30</th>
      <td>fbpca</td>
      <td>1.452501</td>
    </tr>
    <tr>
      <th>30</th>
      <td>sklearn</td>
      <td>1.673517</td>
    </tr>
  </tbody>
</table>
</div>


