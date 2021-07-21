(installation)=
# Installation

## Installation through PyPi
If you already have Python version 3.8.0 or greater, then you can install scarf using `pip`

    pip install scarf

## Installing Python

To use Scarf you need the Python programming language, version 3.8 or upwards, installed.

**Step 1:**

First, check whether you already have Python installed. To do so, you need to open a terminal
window (aka command prompt).

:::{eval-rst}
.. tabs::

  .. tab:: Linux

     Pressing key combination `Ctrl+Alt+T` together works on most Linux distributions.

  .. tab:: Windows

     Press `Win+R` keys on your keyboard. Then, type `cmd` and press `Enter`.

  .. tab:: MacOS

     Press `CMD+Space` to open spotlight search, and type `terminal` and hit `RETURN`.

:::

**Step 2:**

Okay, once you have got a terminal window open, type `python --version` and press `ENTER`:

Now you may see one of the following three kinds of output:

- If your output shows you have `Python 3.8.0` or a more recent version.
  In this case, you are good to go, and you can skip Step 3.
- If you have an earlier version than 3.8, for example 3.5 or 2.7, then see step 3.
- If you see an error containing words `not found` or `not recognized` then most
  likely you do not have a Python version installed on your system. Move to step 3.

**Step 3:**

To install the latest version of Python, we suggest using Miniconda. Download appropriate
version based on your operating system from here:
https://conda.io/miniconda.html (version Python 3.8 or above).

If you are confused about 64-bit or 32-bit see this nice
[post](https://www.techsoup.org/support/articles-and-how-tos/do-i-need-the-32bit-or-64bit)  
Once you have downloaded Miniconda please follow their instructions
[here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
to install it.
Once installed, please confirm that you now have a Python version greater than 3.8.0 by
typing `python --version` in your terminal.

**Step 3.5 (Optional)**

We recommend that you first create an environment that you then install scarf into. 
If you have followed the steps above, and you are using conda you create an environment
by typing ``conda create --name myenv python=3.8``.

This will also install python 3.8 into that environment. One of the benefits of working with
environments is that you minimize the risk of some required package that you are using for
something different (e.g. if you have another package that requires numpy to be of an older
version) is being updated without you wanting it to. This way you can keep the installation
contained.

To activate the environment type `conda activate myenv`. To deactivate it again type
`conda deactivate`.

**Step 4:**

Now, in your terminal, type this to install the latest version of Scarf:
`pip install scarf`
