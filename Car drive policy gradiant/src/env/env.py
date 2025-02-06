import numpy as np
import pygame

class RacingEnv:
    def __init__(self, track_length=500, track_width=10, dt=0.1, is_loop=True, friction=0.05, max_speed=10.0):
        """
        Crée un circuit circulaire avec un modèle Bicycle amélioré.
        track_length : circonférence du cercle.
        track_width  : largeur de la piste.
        dt           : pas de temps de la simulation.
        is_loop      : si True, le circuit est bouclé.
        friction     : coefficient de friction simulant l'inertie.
        max_speed    : vitesse maximale autorisée (m/s).
        """
        self.track_length = track_length
        self.track_width = track_width
        self.dt = dt
        self.is_loop = is_loop
        self.friction = friction
        self.max_speed = max_speed
        self.radius = self.track_length / (2.0 * np.pi)
        
        # Paramètre du modèle Bicycle
        self.length = 2.0  
        self.max_steering_angle = np.radians(30)

        self.reset()
        self._visual_initialized = False

    def reset(self):
        """
        Réinitialise l'environnement et positionne le véhicule sur le cercle idéal.
        """
        self.x = self.radius  
        self.y = 0.0
        self.v = 0.0          
        self.theta = 0.0      
        self.done = False
        self.last_angle = np.arctan2(self.y, self.x)
        return np.array([self.x, self.y, self.v, self.theta])

    def simulate(self, state, action):
        """
        Simule l'évolution du véhicule à partir d'un état donné et d'une action.
        state  : np.array([x, y, v, theta])
        action : (acceleration, steering_angle)
        Retourne (next_state, reward, done, {}).
        """
        x, y, v, theta = state
        acceleration, steering_angle = action
        # Limiter le braquage à la plage autorisée
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # Mise à jour de la vitesse : intégration de l'accélération puis friction exponentielle
        v = max(0.0, v + acceleration * self.dt)
        v = v * np.exp(-self.friction * self.dt)
        if v > self.max_speed:
            v = self.max_speed

        # Calcul de l'angle de dérapage (beta) selon le modèle Bicycle
        beta = np.arctan(0.5 * np.tan(steering_angle))

        # Mise à jour de la position et de l'orientation
        x = x + v * np.cos(theta + beta) * self.dt
        y = y + v * np.sin(theta + beta) * self.dt
        theta = theta + (v / self.length) * np.sin(beta) * self.dt

        # Calcul de la distance par rapport au centre du circuit
        current_radius = np.sqrt(x**2 + y**2)
        deviation = abs(current_radius - self.radius)

        # Vérifier si le véhicule est hors piste
        if current_radius < (self.radius - self.track_width/2) or current_radius > (self.radius + self.track_width/2):
            done = True
            reward = -50 - deviation * 10
        else:
            done = False

            # Bonus de progression
            progress_reward = 5.0 * v

            # Pénalité pour déviation adoucie
            penalty_deviation = -5.0 * deviation

            # Récompense pour l'alignement avec la tangente du circuit
            current_angle = np.arctan2(y, x)
            tangent_angle = current_angle + np.pi/2
            alignment_error = ((theta - tangent_angle + np.pi) % (2*np.pi)) - np.pi
            alignment_reward = -1.0 * abs(alignment_error)

            # Récompense pour le contrôle de la vitesse
            desired_speed = self.max_speed * 0.6
            speed_reward = -abs(v - desired_speed)

            reward = progress_reward + penalty_deviation + alignment_reward + speed_reward

        next_state = np.array([x, y, v, theta])
        return next_state, reward, done, {}

    def render(self, vehicles_states=None, scale_x=2, scale_y=2):
        if not self._visual_initialized:
            pygame.init()
            diameter = 2 * (self.radius + self.track_width/2)
            self.window_width = int(diameter * scale_x)
            self.window_height = int(diameter * scale_y)
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Circuit Circulaire")
            self.clock = pygame.time.Clock()
            self._visual_initialized = True
            self.scale_x = scale_x
            self.scale_y = scale_y
            self.center_x = self.window_width // 2
            self.center_y = self.window_height // 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((200, 200, 200))
        center_color = (255, 255, 0)
        r_px = int(self.radius * self.scale_x)
        pygame.draw.circle(self.screen, center_color, (self.center_x, self.center_y), r_px, 2)
        inner_px = int((self.radius - self.track_width/2) * self.scale_x)
        outer_px = int((self.radius + self.track_width/2) * self.scale_x)
        pygame.draw.circle(self.screen, (0, 0, 0), (self.center_x, self.center_y), inner_px, 3)
        pygame.draw.circle(self.screen, (0, 0, 0), (self.center_x, self.center_y), outer_px, 3)

        if vehicles_states is None:
            vehicles_states = [(self.x, self.y)]
        for (x, y) in vehicles_states:
            x_pix = int(self.center_x + x * self.scale_x)
            y_pix = int(self.center_y - y * self.scale_y)
            pygame.draw.circle(self.screen, (255, 0, 0), (x_pix, y_pix), 6)

        pygame.display.flip()
        self.clock.tick(30)
        pygame.event.pump()  # Ajouté pour s'assurer du traitement des événements