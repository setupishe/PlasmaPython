# for t in tqdm(range(n)):
#     try:
#         move(electrons, nodes, consistency=True)
#         move(ions, nodes, consistency=True)
#     except Exception:
#         print("number of iteration: ", t)
#         break

#     nodes.rho *= 0
#     account_walls(nodes, electrons, left_wall, right_wall)
#     account_walls(nodes, ions, left_wall, right_wall)
#     get_rho(nodes, electrons)
#     get_rho(nodes, ions)
    
#     calc_fields(nodes, h, epsilon, periodic=True)
#     phi_over_time.append(nodes.phi.copy())
#     E_over_time.append(nodes.E.copy())
#     accel(electrons, nodes, L, h, tau)
#     accel(ions, nodes, L, h, tau)
    
#     electrons.denormalise(h, tau)
#     ions.denormalise(h, tau)
#     electron_distrs.append(electrons.v.copy())
#     ion_distrs.append(ions.v.copy())
#     electrons.normalise(h, tau)
#     ions.normalise(h, tau)
    