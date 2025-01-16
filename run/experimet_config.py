import torch


class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # training deepsdf
        self.train_test_ratio = 0.8
        self.hidden_dim = 512
        self.latent_code_dim = 128
        self.xyz_dim = 3
        self.xyz_pos_enc_dim = 3
        self.dropout_prob = 0.001
        self.sample_per_scene = 20000
        self.batch_size = 5
        self.clamping_distance = 1.0
        self.latent_code_regularization = 1e-4
        self.n_epochs = 4002
        self.deepsdf_initial_lr = 1e-3
        self.latent_code_inital_lr = 1e-4
        self.warmup_epoch = 3

        # output training 3d model
        self.train_latent_output_resolution = 256

        # training keyword edit
        self.keyword_reg_epoch = 3000
        self.keyword_reg_initial_lr = 1e-4
        self.keyword_reg_warmup_epoch = 3
        self.keyword_walker_epoch = 3000
        self.keyword_walker_initial_lr = 1e-4

        self.keyword_walker_reg_lambda_ = 1
        self.keyword_walker_content_lambda_ = 5
        self.keyword_walker_batch_size = 5
        self.keyword_walker_warmup_epoch = 3
        self.keyword_attribute = [
            "Voluminous_Smart",
            "Powerful_Delicate",
            "Linear_Curvy",
            "Functional_Decorative",
            "Robust_Flexible",
            "Calm_Dynamic",
            "Realistic_Romantic",
            "Elegant_Cute",
            "Sophisticated_Youthful",
            "Luxurious_Approachable",
            "Formal_Everyday",
            "Strict_Friendly",
            "Uniform_Free",
            "Special_Everyday",
        ]

        # training geometry edit
        self.geometry_reg_epoch = 3000
        self.geometry_reg_initial_lr = 1e-4
        self.geometry_reg_warmup_epoch = 3
        self.geometry_walker_epoch = 3000
        self.geometry_walker_initial_lr = 1e-4

        self.geometry_walker_reg_lambda_ = 1
        self.geometry_walker_content_lambda_ = 8
        self.geometry_walker_batch_size = 5
        self.geometry_walker_warmup_epoch = 3
        self.geometry_attribute = [
            "Roof Length",
            "Cabin Length",
            "Hood Length",
            "L2",
            "Wheelbase",
            "Key Line Base Length",
            "Front Wheel-L Length",
            "Front Overhang",
            "Rear Overhang",
            "RR-OH/FR-OH",
            "H1",
            "Total Height",
            "Nose Height",
            "Nose Slant Amount",
            "Key Line Base Height",
            "Belt Line Height",
            "Front Bumper Lower Edge Height",
            "Rear Bumper Lower Edge Height",
            "Side Sill Lower Edge Height",
            "Key Line Angle",
            "Front Window Inclination Angle",
            "Rear Window Inclination Angle",
            "Overall Width",
            "Cabin Width",
            "Roof Width",
            "Tread Width",
            "Bumper Lower Edge Width",
            "Bumper Upper Edge Width",
            "Roof Thickness",
            "Cabin Thickness",
            "Shoulder Thickness",
            "Hood Thickness",
            "Bumper Thickness",
        ]
 

        # noise data
        self.noise_data = [
            "028_KI_Niro_e_2019",
            "017_FO_Evos_2021",
            "010_CI_C4_Cactus_2015",
            "010_CI_C4_Cactus_2015",
            "056_TO_RAV4_Limited_2018",
            "014_FI_500X_2015",
            "023_HY_Nexo_2019",
            "001_AR_Stelvio_Q4_2017",
            "018_FO_Explorer_ST_2020",
            "002_AR_Tonale_CPT_2019",
            "050_RE_Captur_concept_2020",
            "026_JP_Renegade_Latitude_2014",
            "026_JP_Renegade_Latitude_2014",
            "069_BM_X5_2019",
            "063_VO_XC90_T8_2015",
            "052_TE_Model_X_2016",
            "031_LR_RangeRover_SC_2009",
            "009_CH_Tahoe_RST_2020",
            "088_GC_Yukon_Denali_2021",
            "068_VZ_LadaNiva_Urban_2019",
            "071_BM_iX3_CPT_2018",
            "098_LX_UX_2018",
            "086_GC_Acadia_Denali_2020",
            "099_MR_Levante_2017",
            "113_RE_Arkana_2020",
            "119_TO_bZ4X_2021",
            "037_MZ_CX5_USspec_2012",
        ]
