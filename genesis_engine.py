import numpy as np
import scipy.ndimage as nd

class GenesisEngine:
    def __init__(self, chunk_size=32):
        self.chunk_size = chunk_size
        # Universal RCM Parameters derived from QCD
        self.BETA = 0.09
        self.G = 0.22
        
        # Initialize a chunk with random quantum fluctuations (primordial noise)
        self.field = np.random.rand(chunk_size, chunk_size, chunk_size)

    def calculate_S_gradients(self):
        """
        Calculates the Complexity (C) and Coherence (I) gradients of the 3D field.
        """
        # Coherence (I) corresponds to the Laplacian (diffusion/smoothing)
        laplacian = nd.laplace(self.field)
        
        # Complexity (C) corresponds to the squared gradient (structure formation)
        grad_x, grad_y, grad_z = np.gradient(self.field)
        gradient_squared = grad_x**2 + grad_y**2 + grad_z**2
        
        return laplacian, gradient_squared

    def evolve_field(self, steps=50, dt=0.01):
        """
        Evolves the primordial field using the S-Functional until it reaches 
        a metastable structural attractor (terrain forms).
        """
        for _ in range(steps):
            laplacian, gradient_squared = self.calculate_S_gradients()
            
            # The core RCM field equation:
            # d_rho/dt = Diffusion + Beta*(Complexity) + Gravity(Coherence)
            # (Gravity/Advection term simplified for initial terrain test)
            d_rho = laplacian + (self.BETA * gradient_squared) - (self.G * self.field)
            
            # Update the field
            self.field += d_rho * dt

        return self.field

    def quantize_to_voxels(self):
        """
        Collapses the continuous S-field into discrete block IDs using beta-sectorization.
        """
        voxel_chunk = np.zeros_like(self.field, dtype=int)
        
        # Sector 1: High Persistence / Low Density -> Empty Space (Air = 0)
        voxel_chunk[self.field < 0.3] = 0
        
        # Sector 2: Medium Persistence -> Soft boundaries (Soil/Flora = 1)
        voxel_chunk[(self.field >= 0.3) & (self.field < 0.6)] = 1
        
        # Sector 3: High Density Coherence -> Solid structure (Stone/Metals = 2)
        voxel_chunk[self.field >= 0.6] = 2
        
        return voxel_chunk
