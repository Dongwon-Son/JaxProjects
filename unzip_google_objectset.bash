for i in obj_mesh_zip/*.zip
do tmp=${i%.*} && bd=obj_mesh/${tmp#*/} && unzip $i -d $bd && cp $bd/materials/textures/texture.png $bd/meshes/
done

