import math

import hotspice


## OPTION FOR IP_PINWHEEL WITH PERPFIELDINPUTTER:
# mm = hotspice.ASI.IP_Pinwheel(300e-9, 21, E_B=0.5*55e-3*800e3*2e-22, Msat=800e3, V=2e-22, pattern="uniform")
# mm.add_energy(hotspice.ZeemanEnergy())
# mm.params.UPDATE_SCHEME = "Metropolis"
# datastream = hotspice.io.RandomBinaryDatastream()
# inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=46e-3, angle=7*math.pi/180, relax=False, frequency=0, n=10)
# outputreader = hotspice.io.RegionalOutputReader(5, 5, mm)  # Uses mm, so can't be reused
# dist_exp = hotspice.experiments.IODistanceExperiment(inputter, outputreader, mm)


## OPTION FOR OOP_SQUARE WITH RELEVANT AFM INPUTTERS:
E_B = [ # From the best sample of Sweep20221125115428
			[1.8007466633741765e-17, 1.6245163612398525e-17, 1.7374044918929746e-17, 1.7508135310548484e-17, 1.851384920364391e-17, 1.6975154076810403e-17, 1.8110894586025058e-17, 1.76898478482922e-17, 1.67887105674521e-17, 1.6905627682011854e-17, 1.8334580539067442e-17, 1.6977831919957423e-17, 1.6416922018024448e-17, 1.713045600542325e-17, 1.6890958791425895e-17, 1.946228291666154e-17, 1.7190639548186293e-17, 1.9540009439376437e-17, 1.7480365071771482e-17, 1.7530048473813504e-17],
			[1.670265552382011e-17, 1.6621187852554793e-17, 1.704489456508279e-17, 1.8937401744384293e-17, 1.6963705951832466e-17, 1.8149621544473573e-17, 1.7418387485242562e-17, 1.9129073951512296e-17, 1.6600160946780665e-17, 1.8133676139074982e-17, 1.7377727004974144e-17, 1.750205105069688e-17, 1.567953631752091e-17, 1.6843181303609157e-17, 2.0378133216816462e-17, 1.6711649292627382e-17, 1.880238344794802e-17, 1.8497475798262947e-17, 1.782533277884916e-17, 1.6283359384565917e-17],
			[1.761978856379328e-17, 1.6976818503622753e-17, 1.7530340086553324e-17, 1.739740899390335e-17, 1.733056955198948e-17, 1.8079200789271074e-17, 1.856462705518591e-17, 1.756550253556944e-17, 1.9403717303718142e-17, 1.7923011672787528e-17, 1.7442312172979224e-17, 1.7881120384633847e-17, 1.6100103927931853e-17, 1.680726544068153e-17, 1.6537604229533815e-17, 1.6598662296779975e-17, 1.8457842622382422e-17, 1.7494515778468947e-17, 1.6182746475200924e-17, 1.7785035461057018e-17],
			[1.552563698680075e-17, 1.6911303581121905e-17, 1.68716976597384e-17, 1.7380800145098856e-17, 1.7269975783045897e-17, 1.8951588163541913e-17, 1.8488831405051058e-17, 1.8933232337096267e-17, 1.7005094108363754e-17, 1.6099481911122514e-17, 1.922823319099712e-17, 1.8190330408832965e-17, 1.717915110291085e-17, 1.956268548616757e-17, 1.7296418384297686e-17, 1.7179215569698655e-17, 1.758298962166072e-17, 1.885446593508544e-17, 1.7880370264723057e-17, 1.768051426765934e-17],
			[1.849629907747658e-17, 1.7290305215129306e-17, 1.9224562370213075e-17, 1.683115601172631e-17, 1.8196853704716174e-17, 1.855733808181383e-17, 1.6977718800665274e-17, 1.7318021781602187e-17, 1.736056690313508e-17, 1.837129967088004e-17, 1.7828036754903123e-17, 1.737196886604175e-17, 1.7043841783027792e-17, 1.7061575392571625e-17, 1.8980276961114813e-17, 1.870985081801418e-17, 1.6338223880779466e-17, 1.65909528175994e-17, 1.7256745968462698e-17, 1.753157558990348e-17],
			[1.6969316261062983e-17, 1.7361710686845016e-17, 1.7619102862384086e-17, 1.8946333754349538e-17, 1.7260516144419653e-17, 1.7347926841575142e-17, 1.82658705553522e-17, 1.867760824908478e-17, 1.5600846211650466e-17, 1.7067099406939903e-17, 1.8252154756677198e-17, 1.734621095096366e-17, 1.588945653821335e-17, 1.792236238890808e-17, 1.7712183608753312e-17, 1.7488217226592516e-17, 1.883696883105108e-17, 1.8904601732905107e-17, 1.697212757885307e-17, 1.799352259301197e-17],
			[1.8571661982451763e-17, 1.7363468277772003e-17, 1.6028875157635965e-17, 1.8059267920956603e-17, 1.6000691513677925e-17, 1.7569043993715706e-17, 1.7594722461286306e-17, 1.7059492233158347e-17, 1.8488651311024017e-17, 1.8334953544614445e-17, 1.8547439724920815e-17, 1.74767743880935e-17, 1.8230453683094276e-17, 1.6398815504478706e-17, 1.6605349677640822e-17, 1.813947476879855e-17, 1.7953282428814404e-17, 1.8091485991961458e-17, 1.7464450782009135e-17, 1.551744522986663e-17],
			[1.681681828801653e-17, 1.7924773290405218e-17, 1.761501943993476e-17, 1.768489077276252e-17, 1.773293839121124e-17, 1.710765413351832e-17, 1.7284816940742394e-17, 1.9280170190945384e-17, 1.74478296238529e-17, 1.6601853163085055e-17, 1.677063723915689e-17, 1.7677242894408357e-17, 1.900060457839344e-17, 1.810030514493926e-17, 1.7315503250807e-17, 1.8531698951910087e-17, 1.843423741059599e-17, 1.7471254462102924e-17, 1.8590029395854137e-17, 1.5919994261104767e-17],
			[1.7045038671472136e-17, 1.8491453412343275e-17, 1.8004567022097346e-17, 1.8322841781140126e-17, 1.6343806631419504e-17, 1.7811232362999648e-17, 1.7531277888030476e-17, 1.692435762638024e-17, 1.8345361963274862e-17, 1.8075464203630836e-17, 1.5456625916582382e-17, 1.7527052333194273e-17, 1.7731718537487663e-17, 1.58681774769516e-17, 1.865681715595691e-17, 1.9462917557357902e-17, 1.672950259322132e-17, 1.7645900237808393e-17, 1.519265921720632e-17, 1.6849871882282727e-17],
			[1.835818423430919e-17, 1.8453603095584207e-17, 1.847915502615634e-17, 1.7642007469418896e-17, 1.794364223469054e-17, 1.8809383082435245e-17, 1.7663455498055958e-17, 1.834985249904103e-17, 1.826763716849872e-17, 1.5680092468106565e-17, 1.5880748020359776e-17, 1.7340702341495527e-17, 1.76149975353933e-17, 1.8184279772275628e-17, 1.79145100377804e-17, 1.9358480798449603e-17, 1.8200138293011363e-17, 1.6779452018745567e-17, 1.7086456573230438e-17, 1.8480299206883315e-17],
			[1.763740674528193e-17, 1.8984788672405207e-17, 1.5789972581796817e-17, 1.765407508904717e-17, 1.8632698369046584e-17, 1.7911772118461928e-17, 1.8608398623653184e-17, 1.7742866357753683e-17, 1.7281464300345478e-17, 1.6516597819988885e-17, 1.9081559320150594e-17, 1.786362240342739e-17, 1.7679424266486608e-17, 1.8156768110257177e-17, 1.881238639799162e-17, 1.7270959615661585e-17, 1.5876230901967492e-17, 1.716043282330526e-17, 1.779652008986818e-17, 1.7566100252786392e-17],
			[1.872391397374187e-17, 1.700521663868971e-17, 1.635733503433993e-17, 1.7903430820506366e-17, 1.8572909096676067e-17, 1.785158157025981e-17, 1.6921121035656404e-17, 1.7297411105469172e-17, 1.752964870884878e-17, 1.672918031330722e-17, 1.8212159896650618e-17, 1.73614110291857e-17, 1.8413874879279943e-17, 1.755495744961749e-17, 1.6372317977594693e-17, 1.7285860593580508e-17, 1.7239792004796244e-17, 1.7425882867298317e-17, 1.7167349301408636e-17, 1.8368955134785412e-17],
			[1.7793718344992775e-17, 1.7115794690706316e-17, 1.9148251980778135e-17, 1.852359411734759e-17, 1.7762082790400708e-17, 1.7075245427689575e-17, 1.7909539931539e-17, 1.6754545799597277e-17, 1.8038712837045913e-17, 1.7423732947829613e-17, 1.741701432153929e-17, 1.854208824977563e-17, 1.790385062973478e-17, 1.7295011217159154e-17, 1.6966128035208033e-17, 1.948536755811034e-17, 1.512789745191506e-17, 1.8531915883838433e-17, 1.4794538069855352e-17, 1.696999454219076e-17],
			[1.7929261567807662e-17, 1.7957787500060746e-17, 1.8486439380765516e-17, 1.9373665969241634e-17, 1.7426999975618286e-17, 1.877778563052997e-17, 1.9015983579203803e-17, 1.8166774808630765e-17, 1.7825258039439236e-17, 1.6408928940177607e-17, 1.619562156159509e-17, 1.9549200004239407e-17, 1.86476071790134e-17, 1.7376434165069756e-17, 1.8422487772488058e-17, 1.548897656134304e-17, 1.8635260488325037e-17, 1.8801470442151587e-17, 1.8223288160504142e-17, 1.8111658868773918e-17],
			[1.746115753166942e-17, 1.7554403537213022e-17, 1.7886902409123695e-17, 1.7295976330941427e-17, 1.9479491228179704e-17, 1.9147847392004862e-17, 1.7348516909750324e-17, 1.8134244204249083e-17, 1.8590418473725715e-17, 1.7146393125499597e-17, 1.6822632582160515e-17, 1.736235647509301e-17, 1.8050086240749973e-17, 1.8011296482222907e-17, 1.9174629711520676e-17, 1.789092442618381e-17, 1.73000789522053e-17, 1.5508125032135913e-17, 1.6788920358210244e-17, 1.7373217087170106e-17],
			[1.7835785101540094e-17, 1.5495936233890684e-17, 1.7464924966749955e-17, 1.9513971317597844e-17, 1.6108309425888472e-17, 1.82995695004313e-17, 1.713557418260038e-17, 1.7827848516580752e-17, 1.755091048375882e-17, 1.7837866194521763e-17, 1.656358475215126e-17, 1.7245183793241643e-17, 1.786424354526882e-17, 1.7948983038539e-17, 1.7247375765852992e-17, 1.674661477102899e-17, 1.6038416931831702e-17, 1.6261760383800868e-17, 1.8012167030339217e-17, 1.854377899256701e-17],
			[1.8417321121932164e-17, 1.8872196724767663e-17, 1.8699853860188907e-17, 1.740128914087861e-17, 1.841934761938537e-17, 1.788164403205649e-17, 1.6907294980749415e-17, 1.8873024219274506e-17, 1.7125464024338198e-17, 1.6638226843274864e-17, 1.6618902732329745e-17, 1.87218540044796e-17, 1.8158240634075942e-17, 1.8355766381829928e-17, 1.6747622320251314e-17, 1.6748962254288076e-17, 1.7179929082607078e-17, 1.6779854313309694e-17, 1.8866905692479015e-17, 1.851315025146409e-17],
			[1.6723538594052523e-17, 1.8617003058851448e-17, 1.7488962168615222e-17, 1.7220660412047847e-17, 1.7399002124565987e-17, 1.6396798127862584e-17, 1.803235306722838e-17, 1.7343874025939265e-17, 1.6879383964364902e-17, 1.888400041582025e-17, 1.7477725440494647e-17, 1.8522565559837815e-17, 1.7538655457628228e-17, 1.770186516641861e-17, 1.7501028662093742e-17, 1.798445005619481e-17, 1.7428958782350403e-17, 1.8576763499424473e-17, 1.7621582381093146e-17, 1.6816117224578975e-17],
			[1.7056065444948658e-17, 1.6614034416833806e-17, 1.747435444918973e-17, 1.6786585974568136e-17, 1.631471685844082e-17, 1.8553050034526387e-17, 1.6322825300622663e-17, 1.6600924332271316e-17, 1.8583028908217806e-17, 1.753385973136253e-17, 1.8243794226870213e-17, 1.7742503457857975e-17, 1.6991277703316157e-17, 1.8808691680633994e-17, 1.688014095921362e-17, 1.6329297095338683e-17, 1.5940821406227394e-17, 1.7222193358963654e-17, 1.8245512602944682e-17, 1.6418676659053695e-17],
			[1.5988368570058088e-17, 1.7011596470554296e-17, 1.9012659022328172e-17, 1.9922853827844227e-17, 1.84625551884411e-17, 1.8424892427996178e-17, 1.8753812885702216e-17, 1.7099993082932918e-17, 1.9117633280984247e-17, 1.8809825727175235e-17, 1.749053616645342e-17, 1.7254019723638282e-17, 1.808767929786652e-17, 1.8144466525084063e-17, 1.7272404309809592e-17, 1.7551699610899408e-17, 1.6933229400487784e-17, 1.780647115686128e-17, 1.792717611745127e-17, 1.7824860985118565e-17]
		]
# E_B is the same as from the best (i.e. highest Q) sample of Sweep20221125115428
mm = hotspice.ASI.OOP_Square(a=2.2660680638228574e-07, n=20, E_B=E_B, T=300, moment=2.534014798975711e-16, PBC=False)
mm.add_energy(hotspice.ZeemanEnergy())
mm.params.UPDATE_SCHEME = hotspice.Scheme.NEEL
datastream = hotspice.io.RandomBinaryDatastream()
inputter = hotspice.io.OOPSquareChessFieldInputter(datastream, magnitude=0.071, frequency=1)
outputreader = hotspice.io.OOPSquareChessOutputReader(10, 10)
dist_exp = hotspice.experiments.IODistanceExperiment(inputter, outputreader, mm)

if __name__ == "__main__":
    dist_exp.run(N=50, verbose=True)
    hotspice.plottools.show_m(dist_exp.mm)
    print(dist_exp.output_sequences)
    # dist_exp.calculate_all(input_metric='hamming', output_metric='euclidean')
    dist_exp.calculate_all(input_metric=(lambda x, y: x[-1] != y[-1])) # To see the influence of the final bit
    dist_exp.plot()
