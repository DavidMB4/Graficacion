Para el proyecto 2 os objetos que hice para el mundo (Granja), fueron 2 corrales, un granero y un silo.
Puse texturas para cada uno de los objetos y usando la funcion de load_texture.
~~~
    madera_granero_texture = load_texture('madera_granero.jpg')
    madera_blanca_texture = load_texture('madera_blanca.jpg')
    techo_granero_texture = load_texture("techo_cobertizo_metal.jpg")
    tierra_pasto_texture = load_texture('tierra_pasto.jpg')
    madera_valla_texture = load_texture('madera_valla.jpg')
    lodo_texture = load_texture('lodo.jpg')
    metal_silo_texture = load_texture('metal_silo.jpg')
    metal_silo2_texture = load_texture('metal_silo2.jpg')
~~~

Puse la funcion de cada objeto dentro de draw_scene, pasandoles sus respectivas texturas.
~~~
    position_cobertizo = (-5, 0, 18)
    glPushMatrix()
    glTranslatef(*position_cobertizo)
    glRotatef(90, 0.0, 1.0, 0.0)
    draw_cobertizo(textura_pared, textura_techo, textura_puerta)
    glPopMatrix()
    
    position_granero = (20, 0, 10)
    glPushMatrix()
    glTranslatef(*position_granero)
    draw_granero(madera_granero_texture, madera_blanca_texture, techo_granero_texture)
    glPopMatrix()
    
    position_corral1 = (10, 0, 10)
    glPushMatrix()
    glTranslatef(*position_corral1)
    draw_corral_oveja(tierra_pasto_texture, madera_valla_texture)
    glPopMatrix()
    
    position_corral2 = (30, 0, 10)
    glPushMatrix()
    glTranslatef(*position_corral2)
    draw_corral_cerdo(tierra_pasto_texture, madera_valla_texture, lodo_texture)
    glPopMatrix()
    
    position_silo = (15, 0, 11)
    glPushMatrix()
    glTranslatef(*position_silo)
    draw_silo(metal_silo_texture, metal_silo2_texture)
    glPopMatrix()
~~~

Puse todas estas funciones para el granero:
~~~
def draw_granero(madera_texture, madera_blanca_texture, techo_texture):
    draw_base(madera_texture)
    draw_base2(madera_texture)
    draw_base3(madera_texture)
    draw_techo(techo_texture)
    draw_techo2(techo_texture)
    draw_puerta(madera_blanca_texture, madera_texture)    
    draw_ventana_enfrente(madera_blanca_texture, madera_texture)
    draw_ventana(1, madera_blanca_texture)
    draw_ventana(2, madera_blanca_texture)
    draw_ventana(3, madera_blanca_texture)
    draw_ventana(4, madera_blanca_texture)
    draw_ventana(5, madera_blanca_texture)
    draw_ventana(6, madera_blanca_texture)
~~~

Para el granero en la parte de las bases use glBegin(GL_QUADS), para el techo tambien fueron (GL_QUADS), lo mismo para las ventanas, y para la puerta use (GL_QUEADS) y (GL_TRIANGLES) para hacer los detalles, por ejemplo:
~~~
def puerta_triangulo(texture2):    
    glBindTexture(GL_TEXTURE_2D, texture2)
    glBegin(GL_TRIANGLES)
    glColor3f(0.607, 0.031, 0.031)    
    
    glTexCoord2f(0.0, 0.0); glVertex3f(-2.3, 0.2, 4.15)
    glTexCoord2f(1.0, 0.0); glVertex3f(-0.2, 0.2, 4.15)
    glTexCoord2f(0.5, 1.0); glVertex3f(-1.35, 1.2, 4.15)
    glEnd()
~~~

Para los corrales use (GL_QUADS) donde hice 3 rectangulos, cada uno encima del otro para hacer los tablones, y luego y cubo rectangular para simular los postes que detienen las tablas.
~~~
def draw_corral_cerdo(tierra_pasto_texture, madera_valla_texture, lodo_texture):
    positionsH = [
        (0, 0.3, 0),
        (0, 0.8, 0),
        (0, 1.3, 0),
        (0, 0.3, -8),
        (0, 0.8, -8),
        (0, 1.3, -8)
    ]
    for pos in positionsH:
        glPushMatrix()
        glTranslatef(*pos)
        valla_horizontal(madera_valla_texture)
        glPopMatrix()

    positionsV = [
        (0, 0.3, 0),
        (0, 0.8, 0),
        (0, 1.3, 0),
        (-8, 0.3, 0),
        (-8, 0.8, 0),
        (-8, 1.3, 0)
    ]
    for pos in positionsV:
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(90, 0.0, 1.0, 0.0)
        valla_horizontal(madera_valla_texture)
        glPopMatrix()
        
    positionP = [
        (3.8, 0,  0),
        (3.8, 0,  4),
        (3.8, 0,  -4),
        (-3.8, 0,  0),
        (-3.8, 0,  4),
        (-3.8, 0,  -4),
        (0, 0,  -4)
    ]
    for pos in positionP:
        glPushMatrix()
        glTranslatef(*pos)   
        poste_valla(madera_valla_texture)
        glPopMatrix()
    
    glTranslatef(0.6, 0.03,  0)   
    draw_lodo(lodo_texture)
~~~
Los 2 corrales son similares.

Para el cilo use una funcion para hacer un cilindro grande con gluCylinder() y otra para hacer una esfera usando gluSphere() que seria el techo del silo. Y tiene una entrada hecha con (GL_QUADS) que esta pegada al granero.
~~~
def draw_silo(metal_silo_texture, metal_silo2_texture):
    
    glPushMatrix()
    glTranslatef(5.0, 15.0, -10.0)
    glRotatef(90, 1.0, 0.0, 0.0)
    draw_cylinder(metal_silo2_texture)
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(5.0, 15.0, -10.0)
    glRotatef(90, 1.0, 0.0, 0.0)
    draw_sphere(metal_silo_texture)
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(5.0, 0.0, -5.5)
    entrada_silo(metal_silo2_texture, metal_silo_texture)
    glPopMatrix()
    
    positions = [
        (7, 0, -8.5),
        (7, 0, -8)
    ]
    for pos in positions:
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(180, 0.0, 1.0, 0.0)
        escalera(metal_silo_texture)
        glPopMatrix()
~~~

Para texturizar se hace de esta manera usando glBindTexture(GL_TEXTURE_2D, texture) texxture siendo la textura que se le pasa, y glTexCoord2f(0.0, 0.0) para indicar como se va a poner la textura, los demas objetos se texturizan de forma similar:
~~~
def draw_base(texture):
    """Dibuja el cubo (base de la casa)"""
    glBindTexture(GL_TEXTURE_2D, texture)  # Vincula la textura

    glBegin(GL_QUADS)
    glColor3f(0.717, 0.011, 0.011)  

    # Frente
    glTexCoord2f(0.0, 0.0); glVertex3f(-4, 0, 4)
    glTexCoord2f(1.0, 0.0); glVertex3f(4, 0, 4)
    glTexCoord2f(1.0, 1.0); glVertex3f(4, 4, 4)
    glTexCoord2f(0.0, 1.0); glVertex3f(-4, 4, 4)

    # Atrás
    glTexCoord2f(0.0, 0.0); glVertex3f(-4, 0, -4)
    glTexCoord2f(1.0, 0.0); glVertex3f(4, 0, -4)
    glTexCoord2f(1.0, 1.0); glVertex3f(4, 4, -4)
    glTexCoord2f(0.0, 1.0); glVertex3f(-4, 4, -4)

    # Izquierda
    glTexCoord2f(0.0, 0.0); glVertex3f(-4, 0, -4)
    glTexCoord2f(1.0, 0.0); glVertex3f(-4, 0, 4)
    glTexCoord2f(1.0, 1.0); glVertex3f(-4, 4, 4)
    glTexCoord2f(0.0, 1.0); glVertex3f(-4, 4, -4)

    # Derecha
    glTexCoord2f(0.0, 0.0); glVertex3f(4, 0, -4)
    glTexCoord2f(1.0, 0.0); glVertex3f(4, 0, 4)
    glTexCoord2f(1.0, 1.0); glVertex3f(4, 4, 4)
    glTexCoord2f(0.0, 1.0); glVertex3f(4, 4, -4)

    # Arriba
    glColor3f(0.9, 0.6, 0.3)  # Color diferente para el techo
    glTexCoord2f(0.0, 0.0); glVertex3f(-1, 4, -1)
    glTexCoord2f(1.0, 0.0); glVertex3f(1, 4, -1)
    glTexCoord2f(1.0, 1.0); glVertex3f(1, 4, 1)
    glTexCoord2f(0.0, 1.0); glVertex3f(-1, 4, 1)

    # Abajo
    glColor3f(0.6, 0.4, 0.2)  # Suelo más oscuro
    glTexCoord2f(0.0, 0.0); glVertex3f(-4, 0, -4)
    glTexCoord2f(1.0, 0.0); glVertex3f(4, 0, -4)
    glTexCoord2f(1.0, 1.0); glVertex3f(4, 0, 4)
    glTexCoord2f(0.0, 1.0); glVertex3f(-4, 0, 4)
    glEnd()
~~~

Esto seria el resultado, sin los animales:

![Resultado de mi parte en el proyecto 2]()

El resultado completo esta en el repositorio Proyecto_Graficacion que fue en el repositorio compartido por el equipo.