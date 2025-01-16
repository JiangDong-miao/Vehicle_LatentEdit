import torch
import torch.nn as nn

from .efficient_kan import KAN



class WalkMlpMultiW(nn.Module):
    def __init__(self, attribute_dim, latent_code_dim=128):
        super(WalkMlpMultiW, self).__init__()

        self.linear = nn.Sequential(
            *[
                nn.Linear(latent_code_dim, 2 * latent_code_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(2 * latent_code_dim, 2 * latent_code_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(2 * latent_code_dim, attribute_dim),
            ]
        )

        self.embed = nn.Linear(attribute_dim, latent_code_dim, bias=False)

    def forward(self, latent_codes, delta, lambda_=3):
        out = self.linear(latent_codes)
        out = out / torch.norm(out, dim=1, keepdim=True) * lambda_

        updated_latent_codes = latent_codes + self.embed(delta * out)

        return updated_latent_codes



class WalkKANMulti(nn.Module):
    def __init__(self, attribute_dim, latent_code_dim=128):
        super(WalkKANMulti, self).__init__()

        self.kan1 = KAN(width=[latent_code_dim ,latent_code_dim *2 ,latent_code_dim *2, attribute_dim], grid=5, k=3, seed=0)


        self.embed = KAN(width=[attribute_dim, latent_code_dim],grid=5, k=3, seed=0)

    def forward(self, latent_codes, delta, lambda_=3):
        out = self.kan1(latent_codes)
        out = out / torch.norm(out, dim=1, keepdim=True) * lambda_

        updated_latent_codes = latent_codes + self.embed(delta * out)

        return updated_latent_codes


class WalkEffKANMulti(nn.Module):
    def __init__(self, attribute_dim, latent_code_dim=256):
        super(WalkEffKANMulti, self).__init__()

        self.kan1 = KAN([latent_code_dim ,latent_code_dim *2 ,latent_code_dim *2, attribute_dim])


        self.embed = KAN([attribute_dim, latent_code_dim])

    def forward(self, latent_codes, delta, lambda_=3):
        
        
        out = self.kan1(latent_codes).float()

        out = out / torch.norm(out, dim=1, keepdim=True) * lambda_

 
        updated_latent_codes = latent_codes + self.embed(delta * out).float()



        return updated_latent_codes
    


class WalkEffKAN(nn.Module):
    def __init__(self, attribute_dim, latent_code_dim=256):
        super(WalkEffKAN, self).__init__()

        self.kan1 = KAN([latent_code_dim ,latent_code_dim *2 , attribute_dim],
                        )


        self.embed = KAN([attribute_dim, latent_code_dim],
                        )


    def forward(self, latent_codes, delta, lambda_=3):
        
        
        out = self.kan1(latent_codes).float()

        out = out / torch.norm(out, dim=1, keepdim=True) * lambda_

 
        updated_latent_codes = latent_codes + self.embed(delta * out).float()



        return updated_latent_codes