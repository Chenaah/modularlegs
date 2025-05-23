import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, default='evolution')
    args = parser.parse_args()

    multiprocessing_mode = "ray" # "screen"

    if multiprocessing_mode == "ray":
        from modularlegs.sim.evolution.vae.async_vae_ray import AsyncVAERay
        ga = AsyncVAERay(cfg_name=args.cfg)
    elif multiprocessing_mode == "screen":
        from modularlegs.sim.evolution.vae.async_vae import AsyncVAE
        ga = AsyncVAE(cfg_name=args.cfg)
    ga.run()