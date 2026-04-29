import torch
import torch.nn.functional as F

def crop_to_594x594(tensor: torch.Tensor) -> torch.Tensor:
    h, w = tensor.shape[-2], tensor.shape[-1]

    if h == 596:
        tensor = tensor[1:-1, :]
    elif h != 594:
        raise ValueError(f"Unexpected height: {h}")

    if w == 596:
        tensor = tensor[:, 1:-1]
    elif w != 594:
        raise ValueError(f"Unexpected width: {w}")

    return tensor

fn_denorm2 = lambda x, min, max: (x * (max-min)) + min

# ==========================
# Main function
# ==========================
def compute_momentum_loss(input, label,
                          input_min_gpu, input_max_gpu,
                          output_min_gpu, output_max_gpu,
                          center_idx, n_gpu,
                          dt=3600, theta=35, dxi=200, deta=200,
                          g=9.81, rho=1024, rho_a=1.25):
    '''
    # Global parameters ======
    dt = 3600
    theta = 35
    dxi = 200
    deta = 200
    g = 9.81
    rho = 1024 # in SFINCS rho=1024, rhoa= 1.25 / ref:https://sfincs.readthedocs.io/en/latest/parameters.html
    rho_a= 1.25 
    #Cd = 0.002 # in SFINCS: In SFINCS, Cd is defined as a piecewise function of wind speed: 0.001 (<28 m/s), 0.0025 (28~50 m/s), 0.0015 (>50 m/s)
    '''

    # --- Denormalize ---
    label_denorm = fn_denorm2(label, min=output_min_gpu, max=output_max_gpu)
    input_denorm = fn_denorm2(input, min=input_min_gpu, max=input_max_gpu)

    # --- vars_dict 설정 ---
    vars_dict = {
        "t-1": {"h": label_denorm[center_idx - 1, 0], "u": label_denorm[center_idx - 1, 1], "v": label_denorm[center_idx - 1, 2]},
        "t"  : {"h": label_denorm[center_idx,     0], "u": label_denorm[center_idx,     1], "v": label_denorm[center_idx,     2]},
        "t+1": {"h": label_denorm[center_idx + 1, 0], "u": label_denorm[center_idx + 1, 1], "v": label_denorm[center_idx + 1, 2]},
    }

    h_tm1, u_tm1, v_tm1 = vars_dict["t-1"]["h"], vars_dict["t-1"]["u"], vars_dict["t-1"]["v"]
    h_t,   u_t,   v_t   = vars_dict["t"]["h"],   vars_dict["t"]["u"],   vars_dict["t"]["v"]
    h_tp1, u_tp1, v_tp1 = vars_dict["t+1"]["h"], vars_dict["t+1"]["u"], vars_dict["t+1"]["v"]

    # --- 상수 추출 ---
    B = input_denorm[0,17,:,:] 
    sin_theta = torch.sin(torch.deg2rad(torch.tensor(theta, device="cuda")))
    cos_theta = torch.cos(torch.deg2rad(torch.tensor(theta, device="cuda")))
    U10x = input_denorm[center_idx,13,:,:]
    U10y = input_denorm[center_idx,14,:,:]

    # --- 1) dhu/dt & dv/dt ---
    dhu_dt = crop_to_594x594(((h_tp1 * u_tp1) - (h_tm1 * u_tm1)) / (2 * dt))
    dhv_dt = crop_to_594x594(((h_tp1 * v_tp1) - (h_tm1 * v_tm1)) / (2 * dt))

    # --- 2) d(hu^2)/dx & d(hv^2)/dy ---
    dhu2_dxi = ((h_t*u_t*u_t)[:,2:] - (h_t*u_t*u_t)[:,:-2]) / (2 * dxi)
    dhu2_deta = ((h_t*u_t*u_t)[2:,:] - (h_t*u_t*u_t)[:-2,:]) / (2 * deta)
    dhu2_dx = cos_theta*crop_to_594x594(dhu2_dxi) - sin_theta*crop_to_594x594(dhu2_deta)

    dhv2_dxi = ((h_t*v_t*v_t)[:,2:] - (h_t*v_t*v_t)[:,:-2]) / (2 * dxi)
    dhv2_deta = ((h_t*v_t*v_t)[2:,:] - (h_t*v_t*v_t)[:-2,:]) / (2 * deta)
    dhv2_dy = sin_theta*crop_to_594x594(dhv2_dxi) + cos_theta*crop_to_594x594(dhv2_deta)

    # --- 3) d(hvu)/dy & d(huv)/dx ---
    dhvu_dxi = ((h_t*u_t*v_t)[:,2:] - (h_t*u_t*v_t)[:,:-2]) / (2 * dxi)
    dhvu_deta = ((h_t*u_t*v_t)[2:,:] - (h_t*u_t*v_t)[:-2,:]) / (2 * deta)
    dhuv_dy = sin_theta*crop_to_594x594(dhvu_dxi) + cos_theta*crop_to_594x594(dhvu_deta)
    dhuv_dx = cos_theta*crop_to_594x594(dhvu_dxi) - sin_theta*crop_to_594x594(dhvu_deta)

    # --- 4) gh*d(h+B)/dx ---
    hB = h_t + B
    up, down, left, right = h_t[:-2,1:-1], h_t[2:,1:-1], h_t[1:-1,:-2], h_t[1:-1,2:]
    h_threshold = 0.01
    mask_freesurface = ((up > h_threshold) & (down > h_threshold) & (left > h_threshold) & (right > h_threshold)).int()

    dhB_dxi  = torch.clamp((hB[:,2:] - hB[:,:-2]) / (2 * dxi), -10, 10) # clamp를 통해서 기울기 값이 -10 ~ 10 사이만 유지되도록 해서 physics_loss가 폭발하면서 학습 불안정되는 상황을 최소화 한다. 
    dhB_deta = torch.clamp((hB[2:,:] - hB[:-2,:]) / (2 * deta), -10, 10) 
    ghdhB_dx = g * crop_to_594x594(h_t) * ( cos_theta*crop_to_594x594(dhB_dxi) - sin_theta*crop_to_594x594(dhB_deta) ) * mask_freesurface
    ghdhB_dy = g * crop_to_594x594(h_t) * ( sin_theta*crop_to_594x594(dhB_dxi) + cos_theta*crop_to_594x594(dhB_deta) ) * mask_freesurface

    # --- 5) tau_b/ρ ---
    hmin, hdry = 1e-4, 1e-3
    mask = (h_t >= hdry).to(u_t.dtype)
    speed = torch.sqrt(u_t*u_t + v_t*v_t + 1e-3)
    h13 = (h_t + 1e-3).pow(1/3)
    Cf = g * (n_gpu**2) / h13
    taubx_rho = crop_to_594x594(Cf * speed * u_t * mask)
    tauby_rho = crop_to_594x594(Cf * speed * v_t * mask)

    # --- 6) tau_w/ρ ---
    U10_mag = torch.sqrt(U10x**2 + U10y**2 + 1e-3)
    Cd = torch.where(U10_mag < 28.0, 0.001, 
         torch.where(U10_mag < 50.0, 0.0025, 0.0015))
    tauwx_rho = crop_to_594x594((rho_a / rho) * Cd * U10_mag * U10x)
    tauwy_rho = crop_to_594x594((rho_a / rho) * Cd * U10_mag * U10y)

    # --- Residuals ---
    x_physics_loss = dhu_dt + dhu2_dx + dhuv_dy + ghdhB_dx + taubx_rho - tauwx_rho
    y_physics_loss = dhv_dt + dhuv_dx + dhv2_dy + ghdhB_dy + tauby_rho - tauwy_rho

    # --- Momentum equation loss ---
    x_momentum_loss = x_physics_loss.pow(2).mean()
    y_momentum_loss = y_physics_loss.pow(2).mean()

    # ----------------------------------
    # --- Mass conservation equation ---
    # ----------------------------------

    # --- 1) dh/dt ---
    dh_dt = crop_to_594x594((h_tp1 - h_tm1) / (2 * dt))

    # --- 2) d(Hu)/dx ---
    Hu = (B + h_t)*u_t
    dHu_dxi  = (Hu[:, 2:] - Hu[:, :-2]) / (2 * dxi)
    dHu_deta = (Hu[2:, :] - Hu[:-2, :]) / (2 * deta)
    dHu_dx = (cos_theta * crop_to_594x594(dHu_dxi) - sin_theta * crop_to_594x594(dHu_deta)) * mask_freesurface

    # --- 3) d(Hv)/dy ---
    Hv = (B + h_t)*v_t
    dHv_dxi  = (Hv[:, 2:] - Hv[:, :-2]) / (2 * dxi)
    dHv_deta = (Hv[2:, :] - Hv[:-2, :]) / (2 * deta)
    dHv_dy = (sin_theta * crop_to_594x594(dHv_dxi) + cos_theta * crop_to_594x594(dHv_deta)) * mask_freesurface

    # --- Source term (단위 변환 포함) ---
    precipitation = input_denorm[center_idx, 12, :, :] / 1000 / 3600         # [mm/hr → m/s]
    discharge     = input_denorm[center_idx, 16, :, :] / (dxi * deta) / 3600 # [m³/hr → m/s]
    S_mn = crop_to_594x594(precipitation + discharge)                        # Source term [m/s]

    # --- Residuals ---
    mass_physics_loss = F.relu(dh_dt + dHu_dx + dHv_dy - S_mn)

    # --- Mass equation loss ---
    mass_loss = mass_physics_loss.pow(2).mean()

    return mass_loss, x_momentum_loss, y_momentum_loss

def compute_batch_momentum_loss(input, output,
                                input_min_gpu, input_max_gpu,
                                output_min_gpu, output_max_gpu,
                                n_gpu):
    """
    batch 전체에 대해 physics loss 평균을 계산하는 함수.
    center_idx = 1 ~ batch_size-2 까지 loop를 돌면서 평균냄.
    """

    batch_size = output.shape[0]
    mass_looses, x_losses, y_losses = [], [], []

    for center_idx in range(1, batch_size-1):  # 1 ~ batch_size-2
        mass_loss, x_loss, y_loss = compute_momentum_loss(
            input, output,
            input_min_gpu, input_max_gpu,
            output_min_gpu, output_max_gpu,
            center_idx=center_idx,
            n_gpu=n_gpu
        )
        mass_looses.append(mass_loss)
        x_losses.append(x_loss)
        y_losses.append(y_loss)

    # 텐서로 변환 후 평균
    Mass_loss = torch.stack(mass_looses).mean()
    x_momentum_loss = torch.stack(x_losses).mean()
    y_momentum_loss = torch.stack(y_losses).mean()

    return Mass_loss, x_momentum_loss, y_momentum_loss

def compute_momentum_loss_2d_spatial(input, label,
                                      input_min_gpu, input_max_gpu,
                                      output_min_gpu, output_max_gpu,
                                      center_idx, n_gpu,
                                      dt=3600, theta=35, dxi=200, deta=200,
                                      g=9.81, rho=1024, rho_a=1.25):
    """
    input, label과 scaling 정보, center_idx를 받아서
    x_momentum_loss와 y_momentum_loss의 2D spatial 분포를 반환하는 함수.
    
    Returns:
        x_loss_2d: (594, 594) 형태의 x momentum loss 2D map
        y_loss_2d: (594, 594) 형태의 y momentum loss 2D map
    """
    # --- Denormalize ---
    label_denorm = fn_denorm2(label, min=output_min_gpu, max=output_max_gpu)
    input_denorm = fn_denorm2(input, min=input_min_gpu, max=input_max_gpu)

    # --- vars_dict 설정 ---
    vars_dict = {
        "t-1": {"h": label_denorm[center_idx - 1, 0], "u": label_denorm[center_idx - 1, 1], "v": label_denorm[center_idx - 1, 2]},
        "t"  : {"h": label_denorm[center_idx,     0], "u": label_denorm[center_idx,     1], "v": label_denorm[center_idx,     2]},
        "t+1": {"h": label_denorm[center_idx + 1, 0], "u": label_denorm[center_idx + 1, 1], "v": label_denorm[center_idx + 1, 2]},
    }

    h_tm1, u_tm1, v_tm1 = vars_dict["t-1"]["h"], vars_dict["t-1"]["u"], vars_dict["t-1"]["v"]
    h_t,   u_t,   v_t   = vars_dict["t"]["h"],   vars_dict["t"]["u"],   vars_dict["t"]["v"]
    h_tp1, u_tp1, v_tp1 = vars_dict["t+1"]["h"], vars_dict["t+1"]["u"], vars_dict["t+1"]["v"]

    # --- 상수 추출 ---
    B = input_denorm[0,17,:,:] 
    sin_theta = torch.sin(torch.deg2rad(torch.tensor(theta, device="cuda")))
    cos_theta = torch.cos(torch.deg2rad(torch.tensor(theta, device="cuda")))
    U10x = input_denorm[center_idx,13,:,:]
    U10y = input_denorm[center_idx,14,:,:]

    # --- 1) dhu/dt & dv/dt ---
    dhu_dt = crop_to_594x594(((h_tp1 * u_tp1) - (h_tm1 * u_tm1)) / (2 * dt))
    dhv_dt = crop_to_594x594(((h_tp1 * v_tp1) - (h_tm1 * v_tm1)) / (2 * dt))

    # --- 2) d(hu^2)/dx & d(hv^2)/dy ---
    dhu2_dxi = ((h_t*u_t*u_t)[:,2:] - (h_t*u_t*u_t)[:,:-2]) / (2 * dxi)
    dhu2_deta = ((h_t*u_t*u_t)[2:,:] - (h_t*u_t*u_t)[:-2,:]) / (2 * deta)
    dhu2_dx = cos_theta*crop_to_594x594(dhu2_dxi) - sin_theta*crop_to_594x594(dhu2_deta)

    dhv2_dxi = ((h_t*v_t*v_t)[:,2:] - (h_t*v_t*v_t)[:,:-2]) / (2 * dxi)
    dhv2_deta = ((h_t*v_t*v_t)[2:,:] - (h_t*v_t*v_t)[:-2,:]) / (2 * deta)
    dhv2_dy = sin_theta*crop_to_594x594(dhv2_dxi) + cos_theta*crop_to_594x594(dhv2_deta)

    # --- 3) d(hvu)/dy & d(huv)/dx ---
    dhvu_dxi = ((h_t*u_t*v_t)[:,2:] - (h_t*u_t*v_t)[:,:-2]) / (2 * dxi)
    dhvu_deta = ((h_t*u_t*v_t)[2:,:] - (h_t*u_t*v_t)[:-2,:]) / (2 * deta)
    dhuv_dy = sin_theta*crop_to_594x594(dhvu_dxi) + cos_theta*crop_to_594x594(dhvu_deta)
    dhuv_dx = cos_theta*crop_to_594x594(dhvu_dxi) - sin_theta*crop_to_594x594(dhvu_deta)

    # --- 4) gh*d(h+B)/dx ---
    hB = h_t + B
    up, down, left, right = h_t[:-2,1:-1], h_t[2:,1:-1], h_t[1:-1,:-2], h_t[1:-1,2:]
    h_threshold = 0.01
    mask_freesurface = ((up > h_threshold) & (down > h_threshold) & (left > h_threshold) & (right > h_threshold)).int()
    mask_freesurface = crop_to_594x594(mask_freesurface.float()).int()  # (594, 594) 크기로 변환

    dhB_dxi  = torch.clamp((hB[:,2:] - hB[:,:-2]) / (2 * dxi), -10, 10)
    dhB_deta = torch.clamp((hB[2:,:] - hB[:-2,:]) / (2 * deta), -10, 10)
    ghdhB_dx = g * crop_to_594x594(h_t) * ( cos_theta*crop_to_594x594(dhB_dxi) - sin_theta*crop_to_594x594(dhB_deta) ) * mask_freesurface.float()
    ghdhB_dy = g * crop_to_594x594(h_t) * ( sin_theta*crop_to_594x594(dhB_dxi) + cos_theta*crop_to_594x594(dhB_deta) ) * mask_freesurface.float()

    # --- 5) tau_b/ρ ---
    hmin, hdry = 1e-4, 1e-3
    mask = (h_t >= hdry).to(u_t.dtype)
    speed = torch.sqrt(u_t*u_t + v_t*v_t + 1e-3)
    h13 = (h_t + 1e-3).pow(1/3)
    Cf = g * (n_gpu**2) / h13
    taubx_rho = crop_to_594x594(Cf * speed * u_t * mask)
    tauby_rho = crop_to_594x594(Cf * speed * v_t * mask)

    # --- 6) tau_w/ρ ---
    U10_mag = torch.sqrt(U10x**2 + U10y**2 + 1e-3)
    Cd = torch.where(U10_mag < 28.0, 0.001, 
         torch.where(U10_mag < 50.0, 0.0025, 0.0015))
    tauwx_rho = crop_to_594x594((rho_a / rho) * Cd * U10_mag * U10x)
    tauwy_rho = crop_to_594x594((rho_a / rho) * Cd * U10_mag * U10y)

    # --- Residuals (2D spatial) ---
    x_physics_loss = dhu_dt + dhu2_dx + dhuv_dy + ghdhB_dx + taubx_rho - tauwx_rho
    y_physics_loss = dhv_dt + dhuv_dx + dhv2_dy + ghdhB_dy + tauby_rho - tauwy_rho

    # --- 2D spatial loss (absolute value of residuals) ---
    x_loss_2d = torch.abs(x_physics_loss)
    y_loss_2d = torch.abs(y_physics_loss)

    # --- 각 항별로 개별 저장 (x momentum) ---
    x_abs_dhu_dt = torch.abs(dhu_dt)
    x_abs_dhu2_dx = torch.abs(dhu2_dx)
    x_abs_dhuv_dy = torch.abs(dhuv_dy)
    x_abs_ghdhB_dx = torch.abs(ghdhB_dx)
    x_abs_taubx_rho = torch.abs(taubx_rho)
    x_abs_tauwx_rho = torch.abs(tauwx_rho)

    return x_loss_2d, y_loss_2d, x_abs_dhu_dt, x_abs_dhu2_dx, x_abs_dhuv_dy, x_abs_ghdhB_dx, x_abs_taubx_rho, x_abs_tauwx_rho