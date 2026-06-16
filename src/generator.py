import numpy as np
import cv2

def generate_healthy():
    img = np.zeros((64,64,3), dtype=np.uint8)
    cyto_radius = np.random.randint(20,25)
    
    cv2.circle(
        img,(32,32),cyto_radius,(200,150,100),-1
    )
    
    nucleus_radius = np.random.randint(7,10)
    cv2.circle(
        img,(32,32),nucleus_radius,(60,20,80),-1)
    
    return img

def generate_damage():
    img = np.zeros((64,64,3), dtype=np.uint8)
    
    cyto_radius = np.random.randint(20,25)
    ax = np.random.randint(14,22)
    ay = np.random.randint(8,15)
    
    cv2.ellipse(
        img,
        (32,32),
        (ax,ay),
        np.random.randint(0,360),
        0,
        360,
        (200,150,100),
        -1
    )
    nucleus_radius = np.random.randint(7,10)
    offset_x = np.random.randint(-5,5)
    offset_y = np.random.randint(-5,5)
    ax_n = np.random.randint(7,10)
    ay_n = np.random.randint(7,10)  
    cv2.ellipse(
        img,
        (32+offset_x,32+offset_y),
        (ax_n,ay_n),
        np.random.randint(0,360),
        0,
        360,
        (60,20,80),
        -1
    )
    return img

def generate_damage1():
    img = np.zeros((64,64,3), dtype=np.uint8)
    
    cyto_radius = np.random.randint(20,25)
    ax = np.random.randint(14,22)
    ay = np.random.randint(8,15)
    
    cv2.circle(
        img,
        (32,32),cyto_radius,
        # (ax,ay),
        # np.random.randint(0,360),
        # 0,
        # 360,
        (200,150,100),
        -1
    )
    nucleus_radius = np.random.randint(7,10)
    offset_x = np.random.randint(-5,5)
    offset_y = np.random.randint(-5,5)
    ax_n = np.random.randint(7,10)
    ay_n = np.random.randint(7,10)  
    cv2.ellipse(
        img,
        (32+offset_x,32+offset_y),
        (ax_n,ay_n),
        np.random.randint(0,360),
        0,
        360,
        (60,20,80),
        -1
    )
    return img

def database(
    healthy_count = 4000,
    damage_count1 = 900,
    damage_count2 = 1100,
    save_path="new_liver_dataset6K.npz"
):
    images = []
    labels = []
    
    print("Generating healthy samples...")
    
    for _ in range(healthy_count):
        images.append(generate_healthy())
        labels.append(0)
        
    print("Generating damaged 1st samples...")
    
    for _ in range(damage_count1):
        images.append(generate_damage())
        labels.append(1)
    
    print("Generating damaged 2nd samples...")
        
    for _ in range(damage_count2):
        images.append(generate_damage1())
        labels.append(1)
    
    images = np.array(images)
    labels = np.array(labels)
    
    np.savez(
        save_path,
        images=images,
        labels=labels
    )
    
    print("\nDataset Saved Successfully")
    print("Images Shape :", images.shape)
    print("Labels Shape :", labels.shape)
    print("File :", save_path)
    
if __name__ == "__main__":
        database(
            healthy_count = 4000,
            damage_count1 = 900,
            damage_count2 = 1100
        )
    