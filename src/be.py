import instaloader
import numpy as np
import tensorflow as tf
import pickle
import cv2
import os

from skimage.metrics import structural_similarity as ssim
def predict_profile(username):



    # Učitaj model i scaler
    model_path = r"..\models\instaModel.h5"
    scaler_path = r"..\models\scaler.pkl"
    default_pic_path = r"..\models\default_pic.jpg"
    default_pic2_path = r"..\models\default_pic2.jpg"

    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("Model i scaler uspješno učitani.")

    # Inicijaliziraj instaloader
    L = instaloader.Instaloader()



# Definiraj feature set
    features = [
    "userFollowerCount", "userFollowingCount", "userBiographyLength",
    "userMediaCount", "userHasProfilPic", "userIsPrivate",
    "usernameDigitCount", "usernameLength"
    ]


 # OVDJE UNESI KORISNIKA KOJEG ŽELIŠ PROVJERITI

    try:
        profile = instaloader.Profile.from_username(L.context, username)

    # Preuzimanje profilne slike
        L.download_profile(username, profile_pic_only=True)
        user_dir = os.path.join(os.getcwd(), username)
        profile_pic_filename = next(
            (os.path.join(user_dir, f) for f in os.listdir(user_dir) if f.endswith("_profile_pic.jpg")), None)

        has_profile_pic = 0  # Podrazumijevano nema pravu profilnu sliku
        if profile_pic_filename and os.path.exists(profile_pic_filename):
            user_profile_pic = cv2.imread(profile_pic_filename, cv2.IMREAD_GRAYSCALE)
            default_profile_pic = cv2.imread(default_pic_path, cv2.IMREAD_GRAYSCALE)
            default_profile_pic2 = cv2.imread(default_pic2_path, cv2.IMREAD_GRAYSCALE)

            if user_profile_pic is not None and default_profile_pic is not None and default_profile_pic2 is not None:
                user_profile_pic = cv2.resize(user_profile_pic, (100, 100))
                default_profile_pic = cv2.resize(default_profile_pic, (100, 100))
                default_profile_pic2 = cv2.resize(default_profile_pic2, (100, 100))

                similarity1 = ssim(user_profile_pic, default_profile_pic)
                similarity2 = ssim(user_profile_pic, default_profile_pic2)

                if similarity1 < 0.8 and similarity2 < 0.8:  # Prag sličnosti, može se podesiti
                    has_profile_pic = 1  # Profilna slika nije default

        # Prikupljanje podataka
        data = {
            "userFollowerCount": profile.followers,
            "userFollowingCount": profile.followees,
            "userBiographyLength": len(profile.biography) if profile.biography else 0,
            "userMediaCount": profile.mediacount,
            "userHasProfilPic": has_profile_pic,
            "userIsPrivate": int(profile.is_private),
            "usernameDigitCount": sum(c.isdigit() for c in profile.username),
            "usernameLength": len(profile.username),
        }

    # Konverzija podataka u numpy array
        X_input = np.array([[data[f] for f in features]])
        X_input = scaler.transform(X_input)
        X_input = X_input.reshape(1, 1, X_input.shape[1])

    # Predikcija
        fake_probability = model.predict(X_input)[0][0] * 100

        print("\n==== REZULTATI ====")
        print(f"Korisnik: {username}")
        print(f"Broj pratilaca: {data['userFollowerCount']}")
        print(f"Broj praćenih: {data['userFollowingCount']}")
        print(f"Dužina biografije: {data['userBiographyLength']}")
        print(f"Broj objava: {data['userMediaCount']}")
        print(f"Ima pravu profilnu sliku: {'Da' if has_profile_pic else 'Ne'}")
        print(f"Privatan profil: {'Da' if data['userIsPrivate'] else 'Ne'}")
        print(f"Broj cifara u korisničkom imenu: {data['usernameDigitCount']}")
        print(f"Dužina korisničkog imena: {data['usernameLength']}")
        print(f"Vjerovatnoća da je profil lažan: {round(fake_probability, 2)}%")

        return {
            "profile_data": {
                "username": username,
                "followers": data["userFollowerCount"],
                "followees": data["userFollowingCount"],
                "biography_length": data["userBiographyLength"],
                "mediacount": data["userMediaCount"],
                "has_profile_pic": bool(has_profile_pic),
                "is_private": bool(data["userIsPrivate"]),
                "username_digit_count": data["usernameDigitCount"],
                "username_length": data["usernameLength"]
            },
            "fake_probability": round(fake_probability, 2)
        }

    except Exception as e:
        print(f"Greška: {str(e)}")
