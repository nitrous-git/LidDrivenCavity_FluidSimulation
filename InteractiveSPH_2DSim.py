import pygame
import numpy as np
import numba
import sys

# ----------------------------  
# Pygame & Grid Setup
# ----------------------------
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("SPH Simulation")
clock = pygame.time.Clock()

# Grid configuration
grid_N = 20
cell_width = screen_width // grid_N
cell_height = screen_height // grid_N

# Create the 2D grid (0 = free, 1 = occupied)
grid = [[0 for _ in range(grid_N)] for _ in range(grid_N)]
# Mark some borders as occupied:
for k in range(grid_N):
    grid[grid_N - 1][k] = 1  # bottom row
    grid[k][grid_N - 1] = 1  # right column
    grid[k][0] = 1           # left column

def drawGrid():
    """Draw the grid, marking occupied cells in red and free cells as outlined."""
    for row in range(grid_N):
        for col in range(grid_N):
            x = col * cell_width
            y = row * cell_height
            if grid[row][col] == 1:
                pygame.draw.rect(screen, (200, 0, 0), (x, y, cell_width, cell_height))
            pygame.draw.rect(screen, (122, 122, 122), (x, y, cell_width, cell_height), 1)

# ----------------------------
# SPH Simulation Parameters
# ----------------------------
num_particles = 200  # number of SPH particles
h = 0.02             # smoothing length
m = 1.0              # mass of each particle
rho0 = 100.0         # rest density
B = 10.0             # Tait EOS constant
gamma = 2.0          # Tait exponent
nu = 2.0             # viscosity
g = -9.81            # gravity
dt = 0.01            # time step
bounce = 0.8         # bounce factor for collisions

# Simulation domain boundaries (used for normalization)
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

# ----------------------------
# Particle Class
# ----------------------------
class Particle:
    def __init__(self, pos, vel, half_size):
        """
        pos:  (x, y) in normalized coordinates [0..1]
        vel:  (vx, vy)
        half_size: half of the particle's square hitbox width in normalized coords
        """
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.half_size = half_size  # half-width of the square bounding box

    def checkCollisions(self, grid, grid_N, bounce):
        """
        Checks for collision between this particle's bounding box and any occupied
        grid cells. If collision is found, clamp the position to the nearest side
        and reverse the corresponding velocity component with bounce factor.
        """

        # Current bounding box in normalized coords
        left   = self.pos[0] - self.half_size
        right  = self.pos[0] + self.half_size
        top    = self.pos[1] - self.half_size
        bottom = self.pos[1] + self.half_size

        # Identify which cells this bounding box could overlap
        col_start = max(0, int(left * grid_N))
        col_end   = min(grid_N - 1, int(right * grid_N))
        row_start = max(0, int(top * grid_N))
        row_end   = min(grid_N - 1, int(bottom * grid_N))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if grid[row][col] == 1:  # Occupied cell
                    # Cell boundaries in normalized coords
                    cell_left   = col / grid_N
                    cell_right  = (col + 1) / grid_N
                    cell_top    = row / grid_N
                    cell_bottom = (row + 1) / grid_N

                    # Check if bounding box actually intersects this cell
                    if (right > cell_left and left < cell_right and
                        bottom > cell_top and top < cell_bottom):

                        # Calculate how far we've penetrated each side
                        d_left   = right  - cell_left
                        d_right  = cell_right  - left
                        d_top    = bottom - cell_top
                        d_bottom = cell_bottom - top

                        # Smallest penetration indicates collision side
                        penetration = min(d_left, d_right, d_top, d_bottom)

                        if penetration == d_left:
                            # Colliding with cell's left side
                            self.pos[0] -= d_left
                            self.vel[0] = -bounce * self.vel[0]
                        elif penetration == d_right:
                            # Colliding with cell's right side
                            self.pos[0] += d_right
                            self.vel[0] = -bounce * self.vel[0]
                        elif penetration == d_top:
                            # Colliding with cell's top
                            self.pos[1] -= d_top
                            self.vel[1] = -bounce * self.vel[1]
                        else:
                            # Colliding with cell's bottom
                            self.pos[1] += d_bottom
                            self.vel[1] = -bounce * self.vel[1]

                        # Update bounding box for subsequent checks
                        left   = self.pos[0] - self.half_size
                        right  = self.pos[0] + self.half_size
                        top    = self.pos[1] - self.half_size
                        bottom = self.pos[1] + self.half_size

    def draw(self, screen, screen_width, screen_height):
        """
        Draw the particle as a small circle.
        Converts normalized coordinates [0..1] to screen coords.
        """
        screen_x = int(self.pos[0] * screen_width)
        screen_y = int(self.pos[1] * screen_height)
        pygame.draw.circle(screen, (0, 0, 255), (screen_x, screen_y), 5)

# ----------------------------
# Create Particles
# ----------------------------
particle_size = 0.01  # half-size of each particle's square bounding box
particles = []

np.random.seed(42)
initial_pos_x = np.random.uniform(0.1, 0.4, size=num_particles)
initial_pos_y = np.random.uniform(0.4, 1.0, size=num_particles)
initial_vel_x = np.zeros(num_particles)
initial_vel_y = np.zeros(num_particles)

for i in range(num_particles):
    pos_i = (initial_pos_x[i], initial_pos_y[i])
    vel_i = (initial_vel_x[i], initial_vel_y[i])
    p = Particle(pos_i, vel_i, half_size=particle_size)
    particles.append(p)

# ----------------------------
# Numba-based SPH Functions
# ----------------------------
@numba.njit
def W_2D(dx, dy, h):
    r2 = dx*dx + dy*dy
    return (1.0 / (np.pi * h**2)) * np.exp(-r2 / h**2)

@numba.njit
def gradW_2D(dx, dy, h):
    r2 = dx*dx + dy*dy
    factor = -2.0 / (np.pi * h**4) * np.exp(-r2 / h**2)
    dWx = factor * dx
    dWy = factor * dy
    return dWx, dWy

@numba.njit
def getPairwiseSeparations(posA, posB):
    M = posA.shape[0]
    N = posB.shape[0]
    dx = np.empty((M, N))
    dy = np.empty((M, N))
    for i in range(M):
        for j in range(N):
            dx[i, j] = posA[i, 0] - posB[j, 0]
            dy[i, j] = posA[i, 1] - posB[j, 1]
    return dx, dy

@numba.njit
def getDensity(pos, m, h):
    dx, dy = getPairwiseSeparations(pos, pos)
    M, N = dx.shape
    rho = np.zeros(M)
    for i in range(M):
        for j in range(N):
            rho[i] += m * W_2D(dx[i, j], dy[i, j], h)
    return rho

@numba.njit
def getPressure(rho, rho0, B, gamma):
    M = rho.shape[0]
    p = np.empty(M)
    for i in range(M):
        tmp = B * ((rho[i]/rho0)**gamma - 1.0)
        if tmp < 0.0:
            tmp = 0.0
        p[i] = tmp
    return p

@numba.njit
def getAcc(pos, vel, m, h, rho0, B, gamma, nu, g):
    """
    Compute acceleration for each particle based on:
      1) Tait EOS for pressure
      2) Viscosity
      3) Gravity
    """
    N = pos.shape[0]
    # 1) Density and pressure
    rho = getDensity(pos, m, h)
    P   = getPressure(rho, rho0, B, gamma)
    # 2) Pairwise separations
    dx, dy = getPairwiseSeparations(pos, pos)
    dWx, dWy = gradW_2D(dx, dy, h)

    ax = np.zeros(N)
    ay = np.zeros(N)

    # 3) Pressure force
    for i in range(N):
        for j in range(N):
            # Pressure term = (Pi / (rhoi^2) + Pj / (rhoj^2))
            term = (P[i] / (rho[i]*rho[i]) + P[j] / (rho[j]*rho[j]))
            ax[i] -= m * term * dWx[i, j]
            ay[i] -= m * term * dWy[i, j]

    # 4) Add viscosity and gravity
    for i in range(N):
        ax[i] -= nu * vel[i, 0]
        ay[i] -= nu * vel[i, 1]
        ay[i] -= g  # gravity goes in negative y

    # Return acceleration array
    acc = np.empty((N, 2))
    for i in range(N):
        acc[i, 0] = ax[i]
        acc[i, 1] = ay[i]
    return acc

# ----------------------------
# SPH Step Function
# ----------------------------
def step_sph(particles, dt):
    """
    Implements the leapfrog integration step using the SPH force calculations.
    - Convert each particle's pos, vel to arrays.
    - Compute accelerations with getAcc.
    - Half-kick, drift, collision check, second half-kick.
    """
    N = len(particles)

    # Gather positions, velocities into arrays for SPH
    pos_array = np.zeros((N, 2), dtype=np.float64)
    vel_array = np.zeros((N, 2), dtype=np.float64)
    for i, p in enumerate(particles):
        pos_array[i, :] = p.pos
        vel_array[i, :] = p.vel

    # 1) Half-kick
    acc = getAcc(pos_array, vel_array, m, h, rho0, B, gamma, nu, g)
    vel_array += 0.5 * acc * dt

    # 2) Drift
    pos_array += vel_array * dt

    # 3) After drifting, update each particle, then do collision checks
    for i, p in enumerate(particles):
        p.pos[:] = pos_array[i, :]
        p.vel[:] = vel_array[i, :]
        p.checkCollisions(grid, grid_N, bounce)

    # 4) Second half-kick (using possibly updated positions)
    # Gather again in case collisions changed positions or velocities
    for i, p in enumerate(particles):
        pos_array[i, :] = p.pos
        vel_array[i, :] = p.vel

    acc_new = getAcc(pos_array, vel_array, m, h, rho0, B, gamma, nu, g)
    vel_array += 0.5 * acc_new * dt

    # Store back final velocities
    for i, p in enumerate(particles):
        p.vel[:] = vel_array[i, :]

# ----------------------------
# Main Loop
# ----------------------------
input_modifier = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        # Mouse Action
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            col_index = mouse_x // cell_width
            row_index = mouse_y // cell_height
            if event.button == 1:
                grid[row_index][col_index] = 1
                print(f"Cell occupied: row {row_index}, col {col_index}")
            if event.button == 3:
                grid[row_index][col_index] = 0
                print(f"Cell freed: row {row_index}, col {col_index}")
            if event.button == 2: 
                if input_modifier == 1:
                    # 1) Convert mouse click to normalized coordinates
                    ex = mouse_x / screen_width
                    ey = mouse_y / screen_height

                    # 2) Explosion parameters
                    explosion_radius = 0.5     # in normalized coords
                    explosion_strength = 5.0    # tweak to taste

                    # 3) For each particle, check distance to explosion center and apply impulse
                    for p in particles:
                        dx = p.pos[0] - ex
                        dy = p.pos[1] - ey
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist < explosion_radius and dist > 1e-10:
                            # Normalized direction (dx,dy)
                            nx = dx / dist
                            ny = dy / dist

                            # Option: scale impulse by how close the particle is to center
                            # e.g. more push if closer. This yields a linear falloff.
                            falloff = (1.0 - dist / explosion_radius)

                            # Add an outward velocity impulse
                            p.vel[0] += explosion_strength * falloff * nx
                            p.vel[1] += explosion_strength * falloff * ny
                            print("Explosion at", ex, ey)
                if input_modifier == 2:
                    normalized_x = mouse_x / screen_width
                    normalized_y = mouse_y / screen_height
                    # Middle mouse button: create a new particle
                    new_particle = Particle(
                        pos=[normalized_x, normalized_y],
                        vel=[0.0, 0.0],
                        half_size=particle_size   # same hitbox half-size as others
                    )
                    particles.append(new_particle)
                    print(len(particles))
                    print(f"New particle created at x={col_index:.2f}, y={row_index:.2f}") 
                    
        # Keyboard Action
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: 
                input_modifier = 1     
                print(f"Input modifier = {input_modifier}")         
            if event.key == pygame.K_2: 
                input_modifier = 2
                print(f"Input modifier = {input_modifier}")                 


    # Update simulation (SPH + collisions)
    step_sph(particles, dt)

    # Render everything
    screen.fill((0, 0, 0))
    drawGrid()
    # Draw each particle via its own draw() method
    for p in particles:
        p.draw(screen, screen_width, screen_height)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
