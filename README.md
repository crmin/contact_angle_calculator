# Contact Angle Calculator

## Web Demo Setup
### Prepare
1. Install docker
1. Build docker
    ```
    docker build --tag contact_angle_calculator .
    ```

### Start
```
docker run -p 51273:51273 -d --name=contact_angle contact_angle_calculator
```
You can use web demo after access to `http://localhost:51273`

### Usage
Usage wrote in demo page

## Web Demo Setup (without docker)
### Prepare
1. Install python (version >= 3.8)
    1. (Optional) install virtualenv
    1. (Optional) Make virtual environment using virtualenv
1. Install dependency `pip install -r requirements.txt`

### Start
Execute webserver:
```
python app.py
```
You can use web demo after access to `http://localhost:51273`

If you want to change port number of demo page, open `app.py` and change port number at bottom of code.
```python
app.run(host='0.0.0.0', port='<CHANGE_THIS_NUMBER>')
```

## Command Line Processing
### Prepare
1. Install python (version >= 3.8)
    1. (Optional) install virtualenv
    1. (Optional) Make virtual environment using virtualenv
1. Install dependency `pip install -r requirements.txt`

### Prepare Image
Before execute calculator code, you should decorate image.
1. Prepare contact angle image.
1. Draw BLUE bordered ellipse over liquid drop.
    * You can use any blue color but, reference color is #41719c
1. Draw RED line over surface boundary.
    * You can use any red color but, reference color is #ff0000

![decorated image example](https://user-images.githubusercontent.com/8157830/111880792-d2662d00-89f0-11eb-9b75-02133837f225.png)

### Usage
```
python run.py <image_file_path>
```

# Used Open Sources
* [D2Coding Font](https://github.com/naver/d2codingfont), [Open Font License](https://github.com/naver/d2codingfont/wiki/Open-Font-License)

# LICENSE
[Beerware](https://github.com/crmin/surface_tension_angle/blob/master/LICENSE)