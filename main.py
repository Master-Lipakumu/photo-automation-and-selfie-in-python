import cv2 
#importer cv2 de opencv-python

import datetime
#importer date time


#utilisation de la webcam pour la capture
cap = cv2.VideoCapture(0)


#récuperation du fichier frontalface dans une variable faces
faces = cv2.CascadeClassifier('frontalface.xml')


#récuperation du fichier smile.xml dans une variable smiles
smiles = cv2.CascadeClassifier('smile.xml')


"""
    lancement de la boucle infinie 
    # tant que tu voie un sourire prend la photo ou le selfie
    # si la touche q est presser arret tout le programe
"""
while True:
#lecture de la webcam 
    _, frame = cap.read()
#copy de de la lecture de la webcam
    original_frame = frame.copy()
#convertion de la lecture de la webcam en gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detection du ou des visages
    face = faces.detectMultiScale(gray, 1.3, 5)
#creation de la boucle des inconnus x y w h dans la variable face
    for (x,y,w,h) in face:
#création d'un cadre rectangle en fonction des inconnus données
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)

        face_R = frame[y:y+h, x:x+w]

        gray_R = frame[y:y+h, x:x+w]
#detection du ou des sourires dans la variable smile
        smile = smiles.detectMultiScale(gray_R, 1.3, 25)
#creation de la boucle des inconnus x1 y1 w1 h1 dans la variable smile
        for (x1, y1, w1, h1) in smile:

            cv2.rectangle(face_R, (x1, y1), (x1+w1, y1+h1), (0,0, 255), 2)
#utilisation du temps actuel année, mois, jour, heure, minute,seconde
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#nom du fichier a sauvegarder en temps réel
            filename = f'selfie-{time_stamp}.png'
#écrit chaque fichier en fonction de la vue originale
            cv2.imwrite(filename, original_frame)
#démarage de la camera
    cv2.imshow('Master Lipakumu cam start', frame)
#arret de la boucle si la touche q est presser 
    if cv2.waitKey(10) == ord('q'):
        break


