# Keras-Tuner-CN
Keras-Tuner：适用于TensorFlow 2.0和keras的超参数调优器 （个人翻译版）

欢迎再知乎阅读此[文章](https://zhuanlan.zhihu.com/p/156139224)

## 背景

随着炼丹（深度学习）技术的日益发展，在工业界和学术界都对上好的仙丹（深度学习模型）趋之若鹜，纷纷不惜招收耗费大量钱财去研究炼丹；同时大量招纳炼丹大师以及炼丹药童。目前能使用的炼丹工具也越来越多，根据炼丹使用的丹炉（使用的框架），可以分为如下的类别：

1. 以美利坚谷歌的tensorflow和脸书的Pytorch等为代表的丹炉，这也目前炼丹界流传最为广泛的炼丹工具；此类丹炉理论上能炼制世上万物，但需要炼丹师自己仔细搭建丹炉（模型结构），配置丹方（调参）以及控制稀有而昂贵的真火（GPU）进行炼制（训练）。根据需要炼制的灵材和灵火纯度差异，少则几刻钟；多则需七七四十九天。炼制时需步步谨慎，若有半点偏差，小则丹药不纯，药效不佳（underfitting）；大则前功尽弃（内存爆炸，停电）或走火入魔（overfitting）。如要称为江湖上人人敬仰的炼丹大师，熟悉此类丹炉为必经之路！
2. 某些炼丹师不懈使用谷歌和脸书的丹炉，他们根据独特的炼丹需求，使用自己铸造的丹炉，例如整个丹炉全采用纯金（C++）打造。虽然造炉辛苦，炉子每一个部分可由自己亲手锻造；但能极大地提高炼丹效率，可谓炼丹界“明教”！
3. 以AutoML和Autokeras等为代表的全自动炼丹炉，这是为了让非专业炼丹的人也能够轻松的炼制出较好的仙丹。这些丹炉里已有一些炼丹大师已经调制好的丹方和炼制秘籍。此类方法优点明显，搭建容易，能快速开始进行炼丹；缺点是目前的丹方（模型）数量有限，同时能炼制的灵材仅限于灵草（图像）和宝石（文本），而目前无法对千奇百怪的灵兽（Graph data）进行炼制。
4. 以NNI (Neural Network Intelligence)和keras-tuner为代表的半自动炼丹炉，可以看做是介于全自动炼丹炉和全手动丹炉之间的工具。此类工具仍需要炼丹者自己搭建丹炉，但能自动进行配置丹方（超差调优），本人认为这是炼丹过程中最耗时的步骤；得到最好的配方后就能更好的炼制的仙丹。

以上背景内容受启发于李沐大神的[炼丹文](https://zhuanlan.zhihu.com/p/23781756) ,而本文主要介绍的keras-tuner就是一种半自动炼丹炉，主要用于超参调优，免去调参之苦。**下面主要内容翻译自[keras-tuner的官方文档](https://keras-team.github.io/keras-tuner/)，同时也补充了些个人注释。在查看本文内容时建议先熟悉一下keras的语法，简单容易上手。** 

[快速开始：30 秒上手 Keras]: https://keras.io/zh/#30-keras	" 完全"



## 安装方法 

keras-tuner的环境要求：

```
- Python 3.6
- TensorFlow 2.0
```

安装最新版本的keras-tuner：

```
pip install -U keras-tuner
```

或者直接从源（source）安装：

```
git clone https://github.com/keras-team/keras-tuner.git
cd keras-tuner
pip install .
```



## 基础用法

下面介绍如何使用随机搜索（random search）为单层全连接神经网络（dense neural network）执行超参数调优。

首选定义一个model-building函数，通过这个函数中的参数hp，你可以进行超参数的采样，例如：

```python
hp.Int('units', min_value=32, max_value=512, step=32)  #该句定义了一个范围内的整数
```

该model-building函数会返回一个构建好的模型（compiled model，指整体模型结构已经搭好，包括损失函数的定义等等，但模型还未经过训练）：

```python
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation='relu'))  #定义了一层全连接层
    model.add(layers.Dense(10, activation='softmax')) #定义了用于输出的softmax层
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
```

接下来举例说明如何定义一个tuner（调参器）。首先应该指定**model-building函数**，**需要要优化的目标的名称**（其中优化目标是最小化还是最大化是根据内置metrics自动推断出来的），**用于测试参数的试验次数 (`max_trials`)**，**每一次试验中需要build和fit（拟合）的模型数量(`executions_per_trial`)**。

```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy', #优化目标
    max_trials=5, 
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')
```

目前可用的tuners有RandomSearch和[Hyperband]（https://openreview.net/pdf?id=ry18Ww5ee） 

**Note：** 每次试验中执行多次的目的是减少结果的方差（variance），因此能够更准确地评估模型的性能。但如果希望更快地获得结果，可以设置executions_per_trial=1(只为每个模型配置进行单轮训练)。

可以通过以下的语句输出搜索空间的总结（a summary of the search space）：

```python
tuner.search_space_summary()
```

然后，开始搜索最佳的超参数配置。调用tuner.search与keras中的model.fit()具有相同的语法（signature）：

```python
tuner.search(x, # 训练样本特征
			y, # 训练样本标签
             epochs=5, 
             validation_data=(val_x, val_y)) #定义验证集
```

下面是在tuner.search中发生的过程: 通过调用model-building函数迭代地构建模型，该函数使用hp对象在超参数空间(search space)中进行搜寻（track）。定义的tuner将逐步搜寻超参空间，同时记录下每个配置下的评估指标（metrics）。

但超参搜索结束时，可以通过以下的语句来取回最佳模型：

```python
models = tuner.get_best_models(num_models=2)
```

或者输出模型结果的总结（a summary of the results）：

```python
tuner.results_summary()
```

还能在文件夹 my_dir/helloworld 中查看本次搜索的详细的日志（logs）和检查点（checkpoints）等等。



## 包含条件超参数的搜索空间

下面使用一个for循环，用于创建了一个可调的层数（a tunable number of layers），其中这些层本身包含一个可调参数unit。

```Python
def build_model(hp):
    model = keras.Sequential() # a plain stack of layers
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
```

此方法可以推广到任何层级的具有相互依赖的参数（any level of parameter interdependency），包括递归。注意所有所有参数命名都应该是唯一的（在上述的例子中，在第i次循环里程序将内部参数命名为 **'units_' + str(i)** ）。



## 使用HyperModel子类代替model-building函数

此方法使得共享和重用HyperModel变得容易。HyperModel子类只需要实现一个build(self, hp)方法：

```Python
from kerastuner import HyperModel

class MyHyperModel(HyperModel):

    def __init__(self, classes):
        self.classes = classes #分类数

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model
    
hypermodel = MyHyperModel(classes=10)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             epochs=5,
             validation_data=(val_x, val_y))
```



## 已经构建好的可调 applications：HyperResNet和HyperXception

这些applications是属于能够立即使用（ready-to-use）的计算机视觉超模型，他们都使用 ` loss="categorical_crossentropy"`   和 `metrics=["accuracy"]` 预先构建完成了。  

```python
from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband

hypermodel = HyperResNet(input_shape=(128, 128, 3), classes=10) #直接使用

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```



## 将搜索空间限制为指定参数

如果您已经有一个现有的hypermodel，并且希望只搜索指定参数(如学习速率)，你可以通过将hyperparameters参数传给tuner constructor，以及设置tune_new_entries=False 来指定那些没有在hyperparameters中列出的参数不应该调优。对于这些参数，将使用默认值。

```Python
from kerastuner import HyperParameters

hypermodel = HyperXception(input_shape=(128, 128, 3), classes=10)

hp = HyperParameters()
# This will override the `learning_rate` parameter with your
# own selection of choices
hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

tuner = Hyperband(
    hypermodel,
    hyperparameters=hp,
    # `tune_new_entries=False` prevents unlisted parameters from being tuned
    tune_new_entries=False, # 只会对hp中的超参数调优
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```



## 参数的默认值

无论何时在model-building函数或hypermodel的build方法中申明（ register ）了一个超参数，都可以指定该超参数的默认值：

```Python
hp.Int('units',
       min_value=32,
       max_value=512,
       step=32,
       default=128)
```

如果你没有手动指定，超参数仍然有一个系统默认值（对于int变量，默认值为min_value）。



## 在hypermodel中固定值

如果你想做与上述相反的事情——在一个hypermodel中，将所有可用参数调优，但只有一个参数除外（例如学习速率）。在hyperparameters参数中，设置一个Fixed项（或者任意数量的Fixed项），同时设置tune_new_entries=True。

```Python
hypermodel = HyperXception(input_shape=(128, 128, 3), classes=10)

hp = HyperParameters()
hp.Fixed('learning_rate', value=1e-4) 

tuner = Hyperband(
    hypermodel,
    hyperparameters=hp,
    tune_new_entries=True,  #除了将learning_rate设定为固定值，模型中其它的所有参数都参与调优
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```



##  覆盖构建参数

如果您有一个hypermodel，但希望更改现有的优化器（optimizer）、损失函数（loss）或评估指标（metrics），那么可以通过以下的操作将这些参数传递给tuner constructor来实现：

```Python
hypermodel = HyperXception(input_shape=(128, 128, 3), classes=10)

tuner = Hyperband(
    hypermodel,
    
    optimizer=keras.optimizers.Adam(1e-3),
    loss='mse',
    metrics=[keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')],
    # 覆盖了原来已经构建好的hypermodel中的optimizer，loss和metrics。
    
    objective='val_precision',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x, y,
             validation_data=(val_x, val_y))
```



## 引用 Keras Tuner

如果您的论文中使用到了 Keras Tuner，可以引用如下（BibTex格式）：

```latex
@misc{omalley2019kerastuner,
	title        = {Keras {Tuner}},
	author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others},
	year         = 2019,
	howpublished = {\url{https://github.com/keras-team/keras-tuner}}
}
```



## 后记

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
两年前接触到深度学习时，就纠结是学tensorflow还是Pytorch时。当时想能够快速上手，同时无意看到了Keras，就被其简洁明了的语法所吸引，有种第一次使用Python的感觉，也符合Keras宣传的标语`Deep learning for humans` （暗示其他框架不是人类用的，手动滑稽）。在2018年，Keras正式被Google收入麾下，作为tensorflow 2.x 的官方API （说明我的当年眼光还可以~），也从原来的多后端支持变为逐渐只支持tensorflow了。祝Keras这个项目能越来越好！

最后，此文是从官方文档翻译而来，由于个人水平有限，错误在所难免，欢迎指正! 有些单词总感觉翻译不太准确，最好还是阅读[官方英文文档](https://github.com/keras-team/keras-tuner) 。向科研大佬们致敬！



## References

1. https://github.com/keras-team/keras-tuner
2. https://autokeras.com/
3. https://keras.io/
4. http://www.blog.google/topics/google-cloud/cloud-automl-making-ai-accessible-every-business/
5. 深度学习·炼丹入门 - 李沐的文章 - 知乎 https://zhuanlan.zhihu.com/p/23781756
6. https://github.com/Microsoft/nni



