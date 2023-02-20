+++
title = "PCA"
menu = "main"
weight = 1
+++

# Principal component analysis

## When to use it?

All your variables are numeric.

## Learning material

- [Principal component analysis -- Hervé Abdi & Lynne J. Williams](https://personal.utdallas.edu/~herve/abdi-awPCA2010.pdf)
- [A Tutorial on Principal Component Analysis](https://arxiv.org/pdf/1404.1100.pdf)

## User guide

If you're using PCA it is assumed you have a dataframe consisting of numerical continuous variables. In this example we're going to be using the [Iris flower dataset](https://www.wikiwand.com/en/Iris_flower_data_set).


```python
import pandas as pd
import prince
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)
X = pd.DataFrame(data=X, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
y = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
X.head()
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
      <th>Sepal length</th>
      <th>Sepal width</th>
      <th>Petal length</th>
      <th>Petal width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



The `PCA` class implements scikit-learn's `fit`/`transform` API. It's parameters have to passed at initialisation before calling the `fit` method.


```python
pca = prince.PCA(
    n_components=2,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)
pca = pca.fit(X)
```

The available parameters are:

- `n_components`: the number of components that are computed. You only need two if your intention is to make a chart.
- `n_iter`: the number of iterations used for computing the SVD
- `rescale_with_mean`: whether to substract each column's mean
- `rescale_with_std`: whether to divide each column by it's standard deviation
- `copy`: if `False` then the computations will be done inplace which can have possible side-effects on the input data
- `engine`: what SVD engine to use (should be one of `['fbpca', 'sklearn']`)
- `random_state`: controls the randomness of the SVD results.

Once the `PCA` has been fitted, it can be used to extract the row principal coordinates as so:


```python
pca.transform(X).head()
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.264703</td>
      <td>0.480027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.080961</td>
      <td>-0.674134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.364229</td>
      <td>-0.341908</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.299384</td>
      <td>-0.597395</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.389842</td>
      <td>0.646835</td>
    </tr>
  </tbody>
</table>
</div>



Each column stands for a principal component, whilst there is one row for each row in the original dataset. You can display these projections with the `plot` method.


```python
pca.plot(X.assign(y=y).set_index('y'), color_by='y')
```





<div id="altair-viz-99cc26e3697143e084ecb89905e2b526"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-99cc26e3697143e084ecb89905e2b526") {
      outputDiv = document.getElementById("altair-viz-99cc26e3697143e084ecb89905e2b526");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"data": {"name": "data-2c86d45dfa83f386b78636875b977c9b"}, "mark": {"type": "circle", "size": 50}, "encoding": {"color": {"field": "y", "type": "nominal"}, "tooltip": [{"field": "y", "type": "nominal"}, {"field": "component 0", "type": "quantitative"}, {"field": "component 1", "type": "quantitative"}], "x": {"axis": {"title": "component 0 \u2014 72.96%"}, "field": "component 0", "scale": {"zero": false}, "type": "quantitative"}, "y": {"axis": {"title": "component 1 \u2014 22.85%"}, "field": "component 1", "scale": {"zero": false}, "type": "quantitative"}}, "selection": {"selector001": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}}, {"data": {"name": "data-08c5dd5142568de6f5e077054f3f8ebc"}, "mark": {"type": "square", "color": "green", "size": 50}, "encoding": {"tooltip": [{"field": "variable", "type": "nominal"}], "x": {"field": "component 0", "scale": {"zero": false}, "type": "quantitative"}, "y": {"field": "component 1", "scale": {"zero": false}, "type": "quantitative"}}}], "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-2c86d45dfa83f386b78636875b977c9b": [{"y": "Setosa", "component 0": -2.264702808807588, "component 1": 0.48002659652098867}, {"y": "Setosa", "component 0": -2.0809611519657674, "component 1": -0.6741335566053526}, {"y": "Setosa", "component 0": -2.3642290538903, "component 1": -0.34190802388467617}, {"y": "Setosa", "component 0": -2.2993842170427077, "component 1": -0.5973945076746754}, {"y": "Setosa", "component 0": -2.3898421663138434, "component 1": 0.6468353829020264}, {"y": "Setosa", "component 0": -2.075630948176511, "component 1": 1.4891775233211653}, {"y": "Setosa", "component 0": -2.4440288351341515, "component 1": 0.04764419763001386}, {"y": "Setosa", "component 0": -2.2328471588720142, "component 1": 0.22314807268959136}, {"y": "Setosa", "component 0": -2.3346404779076213, "component 1": -1.1153276754616657}, {"y": "Setosa", "component 0": -2.184328174933942, "component 1": -0.4690135614023755}, {"y": "Setosa", "component 0": -2.1663101007013212, "component 1": 1.043690653053859}, {"y": "Setosa", "component 0": -2.3261308664426976, "component 1": 0.1330783352392312}, {"y": "Setosa", "component 0": -2.218450898822409, "component 1": -0.7286761653165703}, {"y": "Setosa", "component 0": -2.6331006957652274, "component 1": -0.9615067291701623}, {"y": "Setosa", "component 0": -2.1987406032666885, "component 1": 1.8600571132939294}, {"y": "Setosa", "component 0": -2.2622145316010194, "component 1": 2.68628448511059}, {"y": "Setosa", "component 0": -2.2075876958245906, "component 1": 1.4836093631555711}, {"y": "Setosa", "component 0": -2.190349509192298, "component 1": 0.48883831648632703}, {"y": "Setosa", "component 0": -1.8985719958028409, "component 1": 1.405018794466548}, {"y": "Setosa", "component 0": -2.3433690530749915, "component 1": 1.1278493819084747}, {"y": "Setosa", "component 0": -1.9143229960825674, "component 1": 0.40885570775590563}, {"y": "Setosa", "component 0": -2.2070128431947973, "component 1": 0.9241214267468975}, {"y": "Setosa", "component 0": -2.774344702927331, "component 1": 0.4583436677529149}, {"y": "Setosa", "component 0": -1.8186695286958487, "component 1": 0.08555852628736568}, {"y": "Setosa", "component 0": -2.227163305706638, "component 1": 0.1372544553634268}, {"y": "Setosa", "component 0": -1.951846330900376, "component 1": -0.6256185877766763}, {"y": "Setosa", "component 0": -2.0511513727294144, "component 1": 0.24216355266166656}, {"y": "Setosa", "component 0": -2.1685771746542155, "component 1": 0.5271495253082668}, {"y": "Setosa", "component 0": -2.139563451301331, "component 1": 0.3132178101399514}, {"y": "Setosa", "component 0": -2.2652614931542403, "component 1": -0.33773190376048057}, {"y": "Setosa", "component 0": -2.1401221356479843, "component 1": -0.5045406901415181}, {"y": "Setosa", "component 0": -1.8315947706760276, "component 1": 0.4236950676037853}, {"y": "Setosa", "component 0": -2.6149479358589325, "component 1": 1.7935758561044273}, {"y": "Setosa", "component 0": -2.446177391696513, "component 1": 2.1507278773929226}, {"y": "Setosa", "component 0": -2.109974875318652, "component 1": -0.4602018414370372}, {"y": "Setosa", "component 0": -2.207808899078265, "component 1": -0.20610739768843686}, {"y": "Setosa", "component 0": -2.0451462067542003, "component 1": 0.6615581114631074}, {"y": "Setosa", "component 0": -2.527331913170485, "component 1": 0.592292774190809}, {"y": "Setosa", "component 0": -2.429632575084546, "component 1": -0.9041800403761475}, {"y": "Setosa", "component 0": -2.1697107116306626, "component 1": 0.2688789614354704}, {"y": "Setosa", "component 0": -2.286475143345669, "component 1": 0.4417153876990494}, {"y": "Setosa", "component 0": -1.858122456373572, "component 1": -2.3374151575533464}, {"y": "Setosa", "component 0": -2.5536383956143553, "component 1": -0.4791006901223139}, {"y": "Setosa", "component 0": -1.9644476837637397, "component 1": 0.47232666771926}, {"y": "Setosa", "component 0": -2.1370590058116217, "component 1": 1.1422292620394072}, {"y": "Setosa", "component 0": -2.0697442995918296, "component 1": -0.7110527253858937}, {"y": "Setosa", "component 0": -2.384733165778261, "component 1": 1.120429701984535}, {"y": "Setosa", "component 0": -2.3943763142196324, "component 1": -0.38624687258915713}, {"y": "Setosa", "component 0": -2.2294465479426733, "component 1": 0.9979597643079793}, {"y": "Setosa", "component 0": -2.2038334355191296, "component 1": 0.009216357521276055}, {"y": "Versicolor", "component 0": 1.1017811830529471, "component 1": 0.8629724182621572}, {"y": "Versicolor", "component 0": 0.7313374253960869, "component 1": 0.594614725669423}, {"y": "Versicolor", "component 0": 1.2409793195158303, "component 1": 0.6162976544374967}, {"y": "Versicolor", "component 0": 0.40748305881738345, "component 1": -1.75440398932341}, {"y": "Versicolor", "component 0": 1.075474700609077, "component 1": -0.20842104605096656}, {"y": "Versicolor", "component 0": 0.38868733653566395, "component 1": -0.5932836359900759}, {"y": "Versicolor", "component 0": 0.7465297413291604, "component 1": 0.7730193120985946}, {"y": "Versicolor", "component 0": -0.48732274212564086, "component 1": -1.852429086857574}, {"y": "Versicolor", "component 0": 0.9279016383549442, "component 1": 0.03222607789115267}, {"y": "Versicolor", "component 0": 0.011426188736979351, "component 1": -1.034018275129441}, {"y": "Versicolor", "component 0": -0.11019628000063081, "component 1": -2.654072818536563}, {"y": "Versicolor", "component 0": 0.44069344898307766, "component 1": -0.06329518843800261}, {"y": "Versicolor", "component 0": 0.562108306443177, "component 1": -1.7647243806169446}, {"y": "Versicolor", "component 0": 0.7195618886754955, "component 1": -0.18622460583150685}, {"y": "Versicolor", "component 0": -0.03335470317877291, "component 1": -0.4390032099816254}, {"y": "Versicolor", "component 0": 0.8754071908577369, "component 1": 0.5090639567734071}, {"y": "Versicolor", "component 0": 0.3502516679950821, "component 1": -0.19631173455144518}, {"y": "Versicolor", "component 0": 0.15881004754797023, "component 1": -0.7920957424327217}, {"y": "Versicolor", "component 0": 1.2250936335624296, "component 1": -1.6222438030915016}, {"y": "Versicolor", "component 0": 0.164917899386326, "component 1": -1.3026092302957726}, {"y": "Versicolor", "component 0": 0.7376826487712578, "component 1": 0.3965715619602371}, {"y": "Versicolor", "component 0": 0.47628719094097033, "component 1": -0.41732028121355175}, {"y": "Versicolor", "component 0": 1.2341780976571475, "component 1": -0.933325728799279}, {"y": "Versicolor", "component 0": 0.6328581997098207, "component 1": -0.4163877208891003}, {"y": "Versicolor", "component 0": 0.7026611831361812, "component 1": -0.06341181972480113}, {"y": "Versicolor", "component 0": 0.8742736538812894, "component 1": 0.2507933929006107}, {"y": "Versicolor", "component 0": 1.2565091165418822, "component 1": -0.07725601969587009}, {"y": "Versicolor", "component 0": 1.358405121440631, "component 1": 0.33131168179089654}, {"y": "Versicolor", "component 0": 0.6648003672253938, "component 1": -0.22592785469484472}, {"y": "Versicolor", "component 0": -0.04025861090059656, "component 1": -1.0587185465539088}, {"y": "Versicolor", "component 0": 0.13079517549785913, "component 1": -1.5622718342099673}, {"y": "Versicolor", "component 0": 0.023452688970549352, "component 1": -1.5724755942167041}, {"y": "Versicolor", "component 0": 0.24153827295450983, "component 1": -0.7772563825848422}, {"y": "Versicolor", "component 0": 1.0610946088426128, "component 1": -0.6338432447349478}, {"y": "Versicolor", "component 0": 0.22397877351237916, "component 1": -0.28777351204320345}, {"y": "Versicolor", "component 0": 0.42913911551616063, "component 1": 0.8455822409050766}, {"y": "Versicolor", "component 0": 1.0487280512090866, "component 1": 0.5220517968629411}, {"y": "Versicolor", "component 0": 1.0445313843962771, "component 1": -1.382988719190782}, {"y": "Versicolor", "component 0": 0.06958832111642235, "component 1": -0.21950333464771596}, {"y": "Versicolor", "component 0": 0.2834772382875741, "component 1": -1.3293246390695765}, {"y": "Versicolor", "component 0": 0.2790777760554597, "component 1": -1.120028523742404}, {"y": "Versicolor", "component 0": 0.6245697914985707, "component 1": 0.024923029254011395}, {"y": "Versicolor", "component 0": 0.33653037013143455, "component 1": -0.9884040176703603}, {"y": "Versicolor", "component 0": -0.36218338461938465, "component 1": -2.019237873238611}, {"y": "Versicolor", "component 0": 0.28858623882315626, "component 1": -0.8557303199870668}, {"y": "Versicolor", "component 0": 0.0913606556545047, "component 1": -0.18119212582577623}, {"y": "Versicolor", "component 0": 0.22771686553469925, "component 1": -0.38492008098735475}, {"y": "Versicolor", "component 0": 0.5763882886534777, "component 1": -0.1548735972165598}, {"y": "Versicolor", "component 0": -0.4476670190286128, "component 1": -1.5437920343977558}, {"y": "Versicolor", "component 0": 0.25673058888758377, "component 1": -0.5988517961556701}, {"y": "Virginica", "component 0": 1.8445688677230285, "component 1": 0.8704213123248206}, {"y": "Virginica", "component 0": 1.1578816132057785, "component 1": -0.6988698623306915}, {"y": "Virginica", "component 0": 2.205266791075377, "component 1": 0.5620104770083535}, {"y": "Virginica", "component 0": 1.4401506638275383, "component 1": -0.046987588105808144}, {"y": "Virginica", "component 0": 1.8678122203305367, "component 1": 0.2950448244570177}, {"y": "Virginica", "component 0": 2.7518733356662755, "component 1": 0.8004092010275394}, {"y": "Virginica", "component 0": 0.36701768786072325, "component 1": -1.5615028914765063}, {"y": "Virginica", "component 0": 2.3024394446251955, "component 1": 0.4200655796427743}, {"y": "Virginica", "component 0": 2.0066864676766043, "component 1": -0.7114386535471598}, {"y": "Virginica", "component 0": 2.2597773490125, "component 1": 1.921010376459883}, {"y": "Virginica", "component 0": 1.3641754921860072, "component 1": 0.6927564544903849}, {"y": "Virginica", "component 0": 1.6026786704779292, "component 1": -0.42170044977261845}, {"y": "Virginica", "component 0": 1.8839007017032416, "component 1": 0.41924965060512154}, {"y": "Virginica", "component 0": 1.2601150991975063, "component 1": -1.1622604214064645}, {"y": "Virginica", "component 0": 1.467645201017323, "component 1": -0.44227158737708394}, {"y": "Virginica", "component 0": 1.590077317614565, "component 1": 0.6762448057233178}, {"y": "Virginica", "component 0": 1.4714314611333172, "component 1": 0.25562182447146875}, {"y": "Virginica", "component 0": 2.4263289873157006, "component 1": 2.5566612507954884}, {"y": "Virginica", "component 0": 3.310695583933886, "component 1": 0.01778094932062519}, {"y": "Virginica", "component 0": 1.263766673639826, "component 1": -1.7067453803762676}, {"y": "Virginica", "component 0": 2.0377163014694037, "component 1": 0.9104674096183082}, {"y": "Virginica", "component 0": 0.9779807342494206, "component 1": -0.571764324812993}, {"y": "Virginica", "component 0": 2.8976514907341673, "component 1": 0.41364105959564573}, {"y": "Virginica", "component 0": 1.3332321759732075, "component 1": -0.4818112186494305}, {"y": "Virginica", "component 0": 1.7007338974912165, "component 1": 1.0139218673227888}, {"y": "Virginica", "component 0": 1.95432670585307, "component 1": 1.00777759615345}, {"y": "Virginica", "component 0": 1.1751036315549315, "component 1": -0.3163944723097923}, {"y": "Virginica", "component 0": 1.0209505506957903, "component 1": 0.06434602923956033}, {"y": "Virginica", "component 0": 1.7883499201796644, "component 1": -0.1873612145908304}, {"y": "Virginica", "component 0": 1.86364755332826, "component 1": 0.5622907258861427}, {"y": "Virginica", "component 0": 2.435953727922702, "component 1": 0.2592844331442784}, {"y": "Virginica", "component 0": 2.3049277218317643, "component 1": 2.626323468232375}, {"y": "Virginica", "component 0": 1.8627032197949545, "component 1": -0.17854949462549205}, {"y": "Virginica", "component 0": 1.1141477406864735, "component 1": -0.2929226233357326}, {"y": "Virginica", "component 0": 1.2024733016783893, "component 1": -0.8113152708396695}, {"y": "Virginica", "component 0": 2.798770447578107, "component 1": 0.856803329497103}, {"y": "Virginica", "component 0": 1.576255910194754, "component 1": 1.0685811073208047}, {"y": "Virginica", "component 0": 1.3462921036270612, "component 1": 0.42243061085250644}, {"y": "Virginica", "component 0": 0.9248249165424186, "component 1": 0.017223100452282664}, {"y": "Virginica", "component 0": 1.8520450517676692, "component 1": 0.6761281744365193}, {"y": "Virginica", "component 0": 2.0148104299548746, "component 1": 0.6138856369235728}, {"y": "Virginica", "component 0": 1.9017840902621883, "component 1": 0.6895754942430004}, {"y": "Virginica", "component 0": 1.1578816132057785, "component 1": -0.6988698623306915}, {"y": "Virginica", "component 0": 2.0405582280520917, "component 1": 0.8675206009552258}, {"y": "Virginica", "component 0": 1.9981470959523757, "component 1": 1.0491687471841422}, {"y": "Virginica", "component 0": 1.8705032929564096, "component 1": 0.3869660816657235}, {"y": "Virginica", "component 0": 1.5645804830303267, "component 1": -0.8966868088965273}, {"y": "Virginica", "component 0": 1.5211704996278368, "component 1": 0.26906914427794987}, {"y": "Virginica", "component 0": 1.3727877895140728, "component 1": 1.0112544185267902}, {"y": "Virginica", "component 0": 0.9606560300371272, "component 1": -0.024331668169400845}], "data-08c5dd5142568de6f5e077054f3f8ebc": [{"variable": "Sepal length", "component 0": 0.8901687648612943, "component 1": 0.3608298881130246}, {"variable": "Sepal width", "component 0": -0.46014270644790756, "component 1": 0.8827162691623828}, {"variable": "Petal length", "component 0": 0.9915551834193606, "component 1": 0.02341518837916577}, {"variable": "Petal width", "component 0": 0.9649789606692486, "component 1": 0.06399984704374674}]}}, {"mode": "vega-lite"});
</script>



Each principal component explains part of the underlying of the distribution. The explained inertia represents the percentage of the inertia each principal component contributes. The explained inertia is obtained by dividing the eigenvalues obtained with the SVD by the total inertia.


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
      <td>2.918</td>
      <td>72.96%</td>
      <td>72.96%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.914</td>
      <td>22.85%</td>
      <td>95.81%</td>
    </tr>
  </tbody>
</table>
</div>



Eigenvalues can also be visualized with a [scree plot](https://www.wikiwand.com/en/Scree_plot).


```python
pca.scree_plot()
```





<div id="altair-viz-e9d04beed32743bb896bb011c442e4d3"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-e9d04beed32743bb896bb011c442e4d3") {
      outputDiv = document.getElementById("altair-viz-e9d04beed32743bb896bb011c442e4d3");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-155affbd064e7989a3eeba506b32d5c4"}, "mark": {"type": "bar", "size": 10}, "encoding": {"tooltip": [{"field": "component", "type": "nominal"}, {"field": "eigenvalue", "type": "quantitative"}, {"field": "% of variance", "type": "quantitative"}, {"field": "% of variance (cumulative)", "type": "quantitative"}], "x": {"field": "component", "type": "nominal"}, "y": {"field": "eigenvalue", "type": "quantitative"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-155affbd064e7989a3eeba506b32d5c4": [{"component": "0", "eigenvalue": 2.9184978165319935, "% of variance": 72.96244541329983, "% of variance (cumulative)": 72.96244541329983}, {"component": "1", "eigenvalue": 0.9140304714680692, "% of variance": 22.850761786701728, "% of variance (cumulative)": 95.81320720000156}]}}, {"mode": "vega-lite"});
</script>



You can also obtain the correlations between the original variables and the principal components.


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
    </tr>
    <tr>
      <th>variable</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sepal length</th>
      <td>0.890169</td>
      <td>0.360830</td>
    </tr>
    <tr>
      <th>Sepal width</th>
      <td>-0.460143</td>
      <td>0.882716</td>
    </tr>
    <tr>
      <th>Petal length</th>
      <td>0.991555</td>
      <td>0.023415</td>
    </tr>
    <tr>
      <th>Petal width</th>
      <td>0.964979</td>
      <td>0.064000</td>
    </tr>
  </tbody>
</table>
</div>



You may also want to know how much each observation contributes to each principal component.


```python
pca.row_contributions_.head()
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.011716</td>
      <td>0.001681</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009892</td>
      <td>0.003315</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.012768</td>
      <td>0.000853</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.012077</td>
      <td>0.002603</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.013046</td>
      <td>0.003052</td>
    </tr>
  </tbody>
</table>
</div>



Contributions sum up to 100% for each component.


```python
pca.row_contributions_.sum()
```




    component
    0    1.0
    1    1.0
    dtype: float64



You can also transform row projections back into their original space by using the `inverse_transform` method.


```python
pca.inverse_transform(pca.transform(X)).head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.018949</td>
      <td>3.514854</td>
      <td>1.466013</td>
      <td>0.251922</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.738463</td>
      <td>3.030433</td>
      <td>1.603913</td>
      <td>0.272074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.720130</td>
      <td>3.196830</td>
      <td>1.328961</td>
      <td>0.167414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.668436</td>
      <td>3.086770</td>
      <td>1.384170</td>
      <td>0.182247</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.017093</td>
      <td>3.596402</td>
      <td>1.345411</td>
      <td>0.206706</td>
    </tr>
  </tbody>
</table>
</div>


