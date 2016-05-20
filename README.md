Common python functions and classes imported by various of my other
code.

Place into a global repository and point PYTHONPATH to it using:
```
export PYTHONPATH=$PYTHONPATH:/path/to/js/folder/
```
Or clone it into the respective projects python folder (the python
scripts will always call for example:
```
import js.utils.plot.color
```
So as long as python can find that you should be fine.
