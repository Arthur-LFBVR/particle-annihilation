# Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##############################################################################################
# Paramètres initiaux

n_step = 500        # Nombre d'itération
n_particle_a = 10    # Nombre de particule A
n_particle_b = 12    # Nombre de particule B

proba_x = 0.5   # Probabilité de déplacement dans la direction x
proba_y = 0.5   # Probabilité de déplacement dans la direction x
a = 1           # Longueur de déplacement
col = 1

L = 15          # Taille de la boite

pos_init_A = (L/2, 0)
pos_init_B = (-L/2, 0)

box = True      # Présence de boite 
video = True    # Présence de la vidéo 
save_video = True #Sauvegarder la vidéo        


##############################################################################################
# Fonctions

# Initialisation de la position initiale des particules
def init(part : list, n_part : int, val_x : float, val_y : float):
    for p in range(n_part):
        part[p][0][0] = val_x
        part[p][1][0] = val_y


# Conditions limites
def bound_condition(pos_x : float, pos_y : float):
    if (pos_x > L):
        pos_x -= 2*(pos_x-L)
    if (pos_x < -L):
        pos_x -= 2*(pos_x+L)
    if (pos_y > L):
        pos_y -= 2*(pos_y-L)
    if (pos_y < -L):
        pos_y -= 2*(pos_y+L)
    return pos_x, pos_y


# Gestion des collisions
def colision(t : int, part_A : list, part_B : list, n_part_a : int, n_part_b : int, evol : list):
    list_col = []
    coord = []
    for p_a in range(0, n_part_a):
        dist = []
        for p_b in range(0, n_part_b):
            d = np.linalg.norm(np.array([part_A[p_a][0][t], part_A[p_a][1][t]]) - np.array([part_B[p_b][0][t], part_B[p_b][1][t]]))
            if d <= col :
                dist.append({"dist" : d, "p_b" : p_b})
        if len(dist) != 0:
            list_col.append({"p_a" : p_a, "possible_col" : dist})

    if len(list_col) != 0 :
        while len(list_col[0]["possible_col"]) != 0:
            d_min = list_col[0]["possible_col"][0]["dist"]
            n_b = list_col[0]["possible_col"][0]["p_b"]
            ind = 0
            for j in range(1, len(list_col)):
                for k in range(0, len(list_col[j]["possible_col"])):
                    if list_col[j]["possible_col"][k]["dist"] <= d_min  and list_col[j]["possible_col"][k]["p_b"] == n_b :
                        d_min = list_col[j]["possible_col"][k]["dist"]
                        ind = j
        
            coord.append((list_col[ind]["p_a"], n_b))
            del list_col[ind]
            if len(list_col) != 0:
                for j in range(0, len(list_col)):
                    for k in range(0, len(list_col[j]["possible_col"])):
                        if list_col[j]["possible_col"][k]["p_b"] == n_b:
                            del list_col[j]["possible_col"][k]
                            break
            if len(list_col) == 0:
                break
    for i in range(len(coord)):
        part_A = np.delete(part_A, coord[i][0]-i, 0)
        n_part_a -=1
        part_B = np.delete(part_B, coord[i][1]-i, 0)
        n_part_b -=1
    evol[0].append(n_part_a)
    evol[1].append(n_part_b)
    return part_A, part_B, n_part_a, n_part_b, evol


# Marche aléatoire
def position_update(t : int, particles : list, n_particle : int):
    for p in range(0, n_particle):
            direction = np.random.choice([np.random.uniform(-(np.pi)/2, (np.pi)/2), np.random.uniform(0, np.pi), np.random.uniform((np.pi)/2, (3*np.pi)/2), np.random.uniform(np.pi, 2*np.pi)], \
                                         p=[proba_x/2, proba_y/2, (1-proba_x)/2, (1-proba_y)/2])

            particles[p][0][t] = particles[p][0][t-1] + a*np.cos(direction) # Mise à jour de la position en x
            particles[p][1][t] = particles[p][1][t-1] + a*np.sin(direction) # Mise à jour de la position en y

            if box:
                particles[p][0][t], particles[p][1][t] = bound_condition(particles[p][0][t], particles[p][1][t])
    return particles

# Mise en place des figure
def set_figure(figure_title : str, label_x : str, label_y : str, title : str, box : bool):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(figure_title)

    if box:
        plt.hlines(L, -L, L, color = "black")
        plt.hlines(-L, -L, L, color = "black")
        plt.vlines(L, -L, L, color = "black")
        plt.vlines(-L, -L, L, color = "black")
    
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    return fig, ax

# Mise à jour de la position des particules pour la vidéo
def frame_position_update(t : int, particles : list, n_particle : int, color : str, frames : list, ax):
    for p in range(n_particle):
        x = particles[p][0][t]
        y = particles[p][1][t]
        point, = ax.plot(x, y, marker = "o", color=color)
        frames.append(point)
    return frames


##############################################################################################
# Programme principal


fig1, ax1 = set_figure("Random Walk Elimination", "Temps", "Nombre de particules", "Évolution des particules au cours du temps", False)
if video :
    fig2, ax2 = set_figure("Random Walk Elimination Video", "x", "y", "Évolution des différentes particules dans l’espace", box)
    ax2.axvline(0, color = "black", ls = "--")
    artists = []

evol = [[n_particle_a], [n_particle_b]]

particles_A = np.zeros((n_particle_a, 2, n_step))   # Initialisation du tableau de particule A
particles_B = np.zeros((n_particle_b, 2, n_step))   # Initialisation du tableau de particule B
init(particles_B, n_particle_b, -L/2, 0)
init(particles_A, n_particle_a, L/2, 0)

for t in range(1, n_step):
    print(f"Avancement des calculs : {round((t/n_step)*100, 2)} %")
    particles_A = position_update(t, particles_A, n_particle_a)   # Marche aléatoire pour les particules A
    particles_B = position_update(t, particles_B, n_particle_b)   # Marche aléatoire pour les particules B
    particles_A, particles_B, n_particle_a, n_particle_b, evol = colision(t, particles_A, particles_B, n_particle_a, n_particle_b, evol)    # Vérification des collisions 
    
    if video :
        frames = []
        time_text = ax2.text(0.075, 0.925, f"- Particule A = {n_particle_a}\n- Particule B = {n_particle_b}\n- Temps = {t} s", transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        frames.append(time_text)
        frame_position_update(t, particles_A, n_particle_a, "blue", frames, ax2)
        frame_position_update(t, particles_B, n_particle_b, "red", frames, ax2)
        artists.append(frames)

ax1.plot(evol[0], label = "Particule A", color = "blue")
ax1.plot(evol[1], label = "Particule B", color = "red")

box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0 + box1.height * 0.1, box1.width, box1.height * 0.9])
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)

print("Calculs Termines.")

if video :
    ani = animation.ArtistAnimation(fig=fig2, artists=artists, interval=50, blit=True)
    if save_video :
        ani.save('video/random_walk_elimination_video.mp4', writer='ffmpeg', fps=30)
        print("Video sauvegardee.")

plt.tight_layout()
plt.show()