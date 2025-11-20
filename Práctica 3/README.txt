Debes enviar un único fichero ZIP cuyo nombre sea

p3_gggg_nn_apellido1_apellido2.zip,

donde

gggg: número de grupo (1311, 1312, 1391, etc.). 
nn: número de la pareja con dos dígitos (01, 02, 03, etc.).
surname1: Primer apellido del miembro 1 de la pareja.
surname2: Primer apellido del miembro 2 de la pareja.

Los apellidos de los dos miembros deben aparecer por orden alfabético. 

Ejemplos:

p3_1311_01_delval_sanchez.zip
p3_1392_18_bellogin_suarez.zip


La estructura del ZIP a entregar debe ser:

./
├── p3_01.ipynb
├── p3_02.ipynb
├── p3_03.ipynb
├── p3_04.ipynb
├── predicciones/
    └── DATASETNAME_predicciones.csv
├── data/
    ├── DATASETNAME_construccion.csv
    ├── DATASETNAME_explotacion.csv
    └── DATASETNAME_info.txt

donde DATASETNAME es el conjunto de datos elegido en el apartado 4. Las posibilidades son:

 - pet_adoption
 - gym_members_expertise
 - videogame_esrb_ratings

