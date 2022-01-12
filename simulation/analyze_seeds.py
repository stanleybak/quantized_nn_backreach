'''
Analyze specific seeds based on random number generation
'''

import numpy as np

from acasxu_dubins import State, make_random_input

def main():
    """main entry point"""

    out_of_plane_seeds, in_plane_seeds = get_seeds()

    labels = ['in-plane', 'out-of-plane']
    tau_dots = [0, -1]
    seeds = [in_plane_seeds, out_of_plane_seeds]

    for label, tau_dot, seed_list in zip(labels, tau_dots, seeds):
        max_tau = 0 if tau_dot == 0 else 160

        vowns = []
        vints = []

        for seed in seed_list:
            init_vec, cmd_list, init_velo, tau_init = make_random_input(seed, max_tau=max_tau, intruder_can_turn=False)

            s = State(init_vec, tau_init, tau_dot, init_velo[0], init_velo[1], save_states=False)
            s.simulate(cmd_list)

            assert s.min_dist < 500, f"seed was safe: {seed}"

            # state vector is: x, y, theta, x2, y2, theta2, time
            vown, vint = init_velo

            vowns.append(vown)
            vints.append(vint)

        vowns = np.array(vowns)
        vints = np.array(vints)

        print(f"{label}, max vown: {np.max(vowns)}. mean: {vowns.mean()}, std: {vowns.std()}")
        print(f"{label}, min vint: {np.max(vint)}. mean: {vints.mean()}, std: {vints.std()}")
        print()

def get_seeds():
    """returns unsafe seeds from 150 million sims (with tau_max=160)"""

    # these are seeds from 150 million sims
    out_of_plane_seeds = [41018326, 55117612, 81329587, 91541793, 96442897, 110989437, 111137968,
                          111794074, 126241763, 136374795]

    in_plane_seeds = [47864, 227838, 309490, 326376, 382700, 824118, 868059,
        978478, 1046353, 1109685, 1127486, 1150428, 1247120, 1496252, 1703835,
        1750480, 2085003, 2086371, 2208154, 2408364, 2529927, 2552766, 2652992,
        2657242, 2742931, 2874563, 2892634, 2974731, 2989512, 2991660, 3205260,
        3206744, 3249683, 3376341, 3431024, 3547668, 3651486, 3653537, 3668823,
        3722699, 3761497, 3910734, 4030635, 4106780, 4258413, 4376627, 4419229,
        4466118, 4565159, 4602454, 4605034, 4724156, 4818325, 4869908, 4897948,
        5054535, 5075221, 5169557, 5256616, 5286070, 5339360, 5340350, 5779071,
        5782269, 6016230, 6065708, 6165744, 6224907, 6292517, 6342153, 6385701,
        6414145, 6488068, 6538557, 6626016, 6692306, 6733162, 6828858, 6862444,
        7078340, 7086547, 7178781, 7182406, 7407214, 7551082, 7592680, 7630586,
        7712654, 7740827, 7823929, 7872385, 7891139, 7969820, 8041355, 8216808,
        8293051, 8409039, 8568847, 8631263, 9160953, 9168080, 9437984, 9462487,
        9470130, 9507895, 9659812, 9672378, 9840852, 9886985, 9938225, 10082622,
        10198014, 10221323, 10272777, 10284374, 10311514, 10350983, 10381488,
        10405291, 10461219, 10465706, 10530206, 10602014, 10617749, 10664839,
        10809885, 10845401, 10857476, 10910661, 10937981, 11012339, 11023479,
        11223255, 11357386, 11516559, 11695116, 11737037, 11738430, 11752424,
        11792746, 11880615, 11959220, 11974214, 11978018, 12010933, 12066621,
        12081635, 12200549, 12221038, 12321845, 12361431, 12392027, 12843750,
        12877532, 12894182, 13012707, 13031663, 13100440, 13171138, 13354188,
        13389722, 13573442, 13596362, 13628804, 13670681, 13742618, 13801214,
        13818967, 13863773, 13924697, 13933799, 14128516, 14220167, 14221935,
        14294510, 14382278, 14727763, 14790618, 14911646, 15007496, 15033407,
        15116721, 15131608, 15147832, 15448186, 15580276, 15858795, 15915256,
        15964440, 16037872, 16085144, 16498245, 16564693, 16746180, 17054566,
        17133519, 17138957, 17145887, 17240464, 17459123, 17472429, 17507083,
        17538118, 17603003, 17665338, 17681161, 17877004, 17931056, 17932354,
        17974203, 18027662, 18069943, 18229912, 18314102, 18383882, 18545675,
        18643877, 18670654, 18685691, 18737817, 18766873, 18974821, 19094738,
        19260748, 19261524, 19265504, 19370304, 19470510, 19482093, 19489242,
        19574684, 19627487, 19884829, 19956197, 20028446, 20062121, 20118075,
        20133990, 20168667, 20393068, 20479904, 20515123, 20523038, 20555111,
        20642961, 20684605, 20807973, 20810123, 20853091, 20862914, 21095521,
        21242036, 21378432, 21433642, 21440135, 21568625, 21883617, 22034380,
        22063066, 22315177, 22349863, 22550758, 22564291, 22575581, 22576402,
        22671077, 22934194, 23179145, 23332367, 23337920, 23387617, 23488477,
        23587267, 23694001, 23763924, 23869860, 23870022, 23916566, 23982676,
        24002346, 24196895, 24197829, 24205407, 24418065, 24425416, 24432668,
        24489171, 24861660, 25034613, 25132962, 25240465, 25286609, 25454834,
        25540390, 25606743, 25778573, 25782146, 25834901, 26100951, 26209060,
        26238203, 26302724, 26372474, 26589457, 26654078, 26846330, 26895016,
        26915136, 26964546, 27012895, 27085610, 27131287, 27156292, 27161668,
        27168255, 27197873, 27316444, 27446438, 27477441, 27555919, 27615987,
        27745634, 28131510, 28156134, 28172896, 28206111, 28308718, 28354803,
        28361876, 28367488, 28380126, 28507022, 28585044, 28720481, 28972493,
        28993347, 29038793, 29242343, 29315983, 29448164, 29448537, 29543393,
        29587224, 29886093, 29905489, 30105377, 30230147, 30231797, 30236847,
        30295430, 30301935, 30494968, 30570761, 30670389, 30747900, 30827022,
        31024854, 31043012, 31095253, 31132911, 31133621, 31276749, 31381697,
        31555997, 31560163, 31755913, 31758731, 31782405, 32029448, 32029857,
        32043318, 32093298, 32164062, 32242699, 32278063, 32290752, 32301787,
        32386985, 32521295, 32526678, 32533938, 32624376, 32654391, 32829953,
        32944764, 32988329, 33086182, 33385488, 33489827, 33667417, 33823454,
        33824709, 33851205, 33968181, 34044323, 34106975, 34299923, 34551435,
        34625149, 34689776, 34760746, 34838633, 34886085, 34909640, 34986879,
        35004238, 35414263, 35467024, 35691217, 35715728, 35747044, 35767937,
        35903297, 36046088, 36351017, 36440452, 36452070, 36673002, 36764789,
        36776743, 36848561, 36871182, 36909682, 37057070, 37118324, 37128175,
        37184265, 37229812, 37246303, 37518707, 37557005, 37772963, 37778041,
        38011640, 38061668, 38143596, 38148360, 38152881, 38227037, 38391763,
        38552292, 38713314, 38839817, 38909818, 39295025, 39368505, 39407196,
        39499852, 39651104, 39671851, 39786542, 39860616, 39997032, 40014674,
        40036990, 40066852, 40130298, 40134465, 40324080, 40396154, 40453672,
        40455327, 40612440, 40627631, 40684850, 40713777, 40723116, 41012928,
        41015827, 41343984, 41539238, 41684250, 41684374, 41704917, 41711933,
        42118822, 42215475, 42220019, 42227454, 42403864, 42415827, 42646095,
        42693553, 42904972, 42938282, 43000551, 43079907, 43274173, 43318599,
        43335276, 43384536, 43406485, 43426516, 43431194, 43442566, 43536343,
        43674998, 43749213, 43753243, 43879272, 43980279, 44090801, 44438047,
        44498539, 44566927, 44651374, 44760588, 44827392, 44836909, 44872977,
        44889076, 44911814, 44932775, 44951449, 45012752, 45102573, 45166801,
        45217606, 45385315, 45525624, 45533863, 45539155, 45568529, 45570793,
        45737289, 45789913, 45853441, 45949348, 45967123, 46126987, 46169845,
        46218434, 46280910, 46342130, 46522981, 46527342, 46667733, 46744824,
        47001658, 47125015, 47172838, 47345050, 47369569, 47385292, 47494315,
        47554357, 47580779, 47740657, 47869383, 47873370, 47882049, 47883362,
        47974452, 47999174, 48067336, 48177122, 48270011, 48336166, 48585851,
        48664628, 48757734, 48790603, 48818109, 49048168, 49343337, 49413150,
        49483117, 49735171, 49920564, 49938692, 49983127, 50063156, 50103499,
        50260801, 50317744, 50399220, 50417856, 50503523, 50703962, 50763394,
        50817086, 50818352, 50840556, 50885042, 51070681, 51214438, 51219692,
        51507372, 51629269, 51745633, 51840272, 52078198, 52091416, 52300805,
        52711630, 52890701, 53108412, 53141528, 53252575, 53489621, 53651959,
        53747646, 54044251, 54288540, 54307543, 54510950, 54764918, 54820040,
        55034953, 55059704, 55116638, 55266114, 55353097, 55358381, 55405398,
        55472342, 55519417, 55541679, 55545704, 55917241, 56029064, 56077626,
        56184077, 56193183, 56251830, 56497850, 56531242, 56545214, 56546110,
        56564920, 56569772, 56615167, 56750554, 56994979, 57010961, 57027971,
        57153094, 57199371, 57203721, 57248785, 57266001, 57595570, 57628710,
        57697358, 57742210, 57761566, 57825212, 57968324, 58157295, 58165326,
        58193313, 58345322, 58365091, 58398995, 58462332, 58544917, 58567570,
        58638177, 58698494, 58797965, 58920557, 59054473, 59102846, 59311113,
        59427828, 59463889, 59489261, 59519414, 59529597, 59543033, 59723027,
        60063934, 60100723, 60217020, 60217476, 60264175, 60268582, 60289896,
        60290610, 60315287, 60501901, 60625782, 60627450, 60703468, 60724558,
        60745328, 60776138, 60869570, 60942369, 61034631, 61125735, 61263115,
        61328751, 61345300, 61502181, 61626803, 61632233, 61696270, 61789223,
        61810292, 61905732, 61930908, 61975048, 61984654, 62061238, 62081979,
        62190389, 62268659, 62445370, 62648652, 62758862, 62985085, 63027602,
        63070261, 63126534, 63226130, 63281754, 63312672, 63393405, 63439375,
        63735789, 63857886, 63904949, 63949701, 63982903, 64065315, 64114176,
        64183170, 64440141, 64776957, 64797890, 65302107, 65434783, 65573026,
        65584526, 65663301, 65672583, 65706229, 65718569, 65753372, 65793145,
        65801444, 66354013, 66382819, 66541025, 66666564, 66714763, 66794239,
        66798345, 66827387, 66870085, 66978331, 67278801, 67366111, 67576049,
        67837701, 67838695, 67870387, 67881840, 68002672, 68019457, 68020080,
        68050001, 68068656, 68141260, 68164474, 68263695, 68364046, 68381908,
        68397933, 68402011, 68411736, 68449276, 68628184, 68800258, 68830393,
        68855471, 68951598, 69014365, 69063306, 69168289, 69286292, 69289610,
        69361606, 69825182, 69922660, 70022017, 70122249, 70152551, 70354557,
        70435971, 70767685, 70913311, 70985572, 71328525, 71420484, 71496974,
        71890112, 71919391, 71941804, 72380648, 72614621, 72680148, 72775652,
        72804798, 72814493, 73099178, 73160471, 73178168, 73184491, 73211793,
        73219951, 73272070, 73438644, 73518617, 73612769, 73644016, 73939509,
        74036107, 74049964, 74091263, 74093981, 74216057, 74255253, 74256085,
        74283258, 74341797, 74453088, 74566879, 74567076, 74574522, 74644180,
        74785048, 74840642, 74939542, 74942682, 74958855, 75086994, 75172932,
        75238871, 75298436, 75309385, 75503765, 75620305, 75654932, 75721033,
        75740107, 75936272, 75975945, 76060177, 76158178, 76195009, 76198554,
        76359673, 76399663, 76419460, 76609912, 76997458, 77065609, 77075896,
        77109752, 77161313, 77295475, 77318903, 77383917, 77421585, 77553645,
        77613474, 77649964, 77654933, 77707918, 77733704, 77856613, 77924893,
        77963887, 78001368, 78010364, 78040299, 78195764, 78213808, 78278094,
        78511106, 78548579, 78724260, 78779057, 79027137, 79078963, 79249587,
        79261309, 79451669, 79475468, 79557189, 79678040, 79746955, 79800296,
        79880311, 79929952, 79951971, 79985607, 80101789, 80244686, 80373173,
        80420797, 80527574, 80555189, 80637245, 80958686, 81006122, 81079742,
        81107583, 81179065, 81320334, 81478004, 81530275, 81579124, 81712282,
        82067143, 82081687, 82169928, 82270999, 82371911, 82398489, 82406721,
        82508744, 82588296, 82843490, 82849995, 82883241, 82896581, 82916411,
        82973467, 83492801, 83632598, 83697601, 83985306, 84127968, 84237623,
        84286228, 84317520, 84674682, 84679273, 84813847, 84820019, 84900910,
        85111499, 85183658, 85192227, 85200299, 85377977, 85435294, 85642042,
        85728929, 85833883, 85919596, 85995004, 86006999, 86011987, 86030192,
        86079882, 86082218, 86136036, 86143330, 86176849, 86307007, 86386976,
        86407772, 86655359, 86720566, 86825202, 86846945, 86940314, 87038770,
        87053402, 87072356, 87228122, 87239012, 87320508, 87362246, 87375047,
        87691079, 87698683, 87751425, 87789082, 88080700, 88120170, 88121636,
        88157100, 88237480, 88279098, 88358093, 88421015, 88421105, 88583411,
        88589915, 88763608, 88768552, 88815751, 88935158, 88966649, 89077322,
        89305566, 89536998, 89807994, 89993765, 89996233, 90011059, 90074401,
        90099993, 90254081, 90425961, 90427805, 90451139, 90533613, 90651076,
        90813499, 90974995, 90975488, 91115066, 91221839, 91330884, 91614582,
        91665613, 91724548, 91808033, 91810150, 92004806, 92051698, 92251698,
        92568373, 92713321, 92824847, 92877773, 92956146, 93117549, 93118743,
        93136754, 93150942, 93190447, 93216939, 93234588, 93348716, 93377065,
        93642504, 93644260, 93718631, 94015983, 94021814, 94028746, 94124043,
        94248747, 94442955, 94502809, 94635423, 94905334, 95002788, 95106928,
        95132900, 95211349, 95229307, 95484075, 95935552, 95976417, 96192010,
        96231683, 96263104, 96338064, 96605674, 96699498, 96705637, 96802340,
        96896111, 97115843, 97163608, 97349880, 97357278, 97398017, 97421287,
        97529617, 97662349, 97854819, 98337343, 98386056, 98558803, 98560499,
        98561902, 98599297, 98612270, 98702500, 98723184, 98754976, 98778496,
        98837490, 98958126, 98958255, 99006733, 99101272, 99306611, 99383668,
        99384524, 99525306, 99531652, 99652688, 99726376, 99767619, 99860083,
        99995887, 100068333, 100089419, 100106255, 100203721, 100248746, 100278256,
        100356648, 100397832, 100479227, 100675625, 100747006, 100873100,
        101083505, 101145344, 101387975, 101740422, 101746707, 101840259,
        101867541, 101936297, 101958922, 102164233, 102185511, 102314725,
        102373723, 102409291, 102550153, 102570036, 102591079, 102937558,
        102983581, 103083875, 103155076, 103302040, 103434962, 103650592,
        103831512, 103966640, 104051413, 104176688, 104309733, 104629910,
        105037370, 105059952, 105080257, 105115828, 105151022, 105157039,
        105166257, 105225722, 105371531, 105405352, 105429807, 105468539,
        105549706, 105660672, 105664486, 105760085, 105898323, 105903092,
        105914221, 105972265, 105985025, 106063170, 106182585, 106244481,
        106857257, 106896366, 106932888, 106946186, 106954016, 106989492,
        106998105, 106999650, 107002915, 107061941, 107222500, 107246421,
        107325871, 107442772, 107568188, 107585875, 107685197, 107709402,
        107722606, 107946886, 108085650, 108102888, 108169339, 108185366,
        108197956, 108223477, 108256514, 108283576, 108299480, 108310864,
        108333200, 108367814, 108587713, 108738434, 108908296, 109066758,
        109087120, 109248117, 109268649, 109284663, 109334144, 109484012,
        109519695, 109520317, 109702174, 109797685, 109856441, 109883766,
        109953742, 110259212, 110261179, 110432706, 110499282, 110701228,
        110787558, 110886731, 110904692, 110918338, 110953589, 111024939,
        111066769, 111228415, 111270053, 111330585, 111360442, 111410904,
        111521990, 111603314, 111607373, 111690206, 111705053, 112093678,
        112152519, 112183848, 112245213, 112441261, 112476514, 112531657,
        112610532, 112645348, 112689676, 112696441, 112811270, 112892595,
        113018179, 113179542, 113233409, 113280194, 113527811, 113723150,
        113808243, 113836859, 114032251, 114036031, 114090184, 114096837,
        114270987, 114404507, 114424276, 114500517, 114515723, 114552279,
        114816553, 114817369, 114834512, 114900019, 114928607, 115138981,
        115249160, 115336671, 115640100, 115777819, 115991445, 116207138,
        116249882, 116310133, 116351474, 116374574, 116393053, 116400013,
        116423635, 116452713, 116466042, 116515577, 116560303, 116793477,
        116843332, 116929279, 117110342, 117123296, 117222285, 117264170,
        117266085, 117532800, 117658202, 117662205, 117690343, 117704194,
        117906574, 118004262, 118009925, 118425817, 118446137, 118446458,
        118484232, 118560231, 118720202, 118747952, 118941187, 119087213,
        119160294, 119279318, 119334680, 119429419, 119629373, 119755603,
        119798959, 119847406, 119920020, 119956898, 119971262, 120100370,
        120145569, 120154647, 120274404, 120340262, 120496250, 120592511,
        120718398, 120750526, 120877168, 120942885, 121001192, 121155680,
        121290959, 121381163, 121389963, 121461851, 121509750, 121592369,
        121600983, 121789729, 121790386, 121809489, 121817873, 121946133,
        122077270, 122135301, 122137605, 122173727, 122260185, 122265387,
        122293240, 122299149, 122345400, 122357209, 122427097, 122519241,
        122631894, 122651497, 122671640, 122764633, 122890792, 122891130,
        123058563, 123122927, 123180822, 123221581, 123253693, 123324394,
        123344793, 123439485, 123615565, 123674831, 123823603, 123841296,
        124098673, 124099367, 124142766, 124312648, 124390669, 124416330,
        124468331, 124488054, 124558783, 124612464, 124806402, 124815319,
        124843825, 124878698, 124990056, 125112150, 125264576, 125331747,
        125348952, 125356786, 125460061, 125522450, 125609583, 125667199,
        125701052, 125710420, 125741153, 125771026, 126026166, 126091964,
        126121833, 126228680, 126337939, 126348891, 126498862, 126536324,
        126593353, 126679687, 126739889, 126741175, 126741785, 126867040,
        126945311, 127026498, 127245084, 127396318, 127398350, 127438091,
        127447997, 127448673, 127528628, 127561572, 127695973, 128091590,
        128298894, 128306547, 128425204, 128429500, 128467389, 128505363,
        128610054, 128674897, 128708209, 129156454, 129258202, 129361058,
        129466715, 129467479, 129603626, 129672863, 129861379, 129938560,
        130029671, 130123356, 130141754, 130175227, 130191252, 130237187,
        130241254, 130245933, 130272790, 130331052, 130348290, 130348753,
        130385282, 130526943, 130865684, 130897270, 130909370, 131037922,
        131104631, 131365139, 131378154, 131468815, 131498546, 131547682,
        131652556, 131682468, 131691009, 131707352, 131819941, 131850832,
        131959847, 131960244, 132006556, 132046809, 132052704, 132186677,
        132274293, 132312628, 132567856, 132604229, 132793013, 132837941,
        132997676, 133402467, 133833563, 133846160, 133858464, 133870779,
        133875201, 133876716, 133992231, 134041859, 134047487, 134140363,
        134166312, 134258143, 134373931, 134687316, 134696594, 134749702,
        134916835, 134972409, 135007548, 135094985, 135227345, 135275310,
        135291635, 135335804, 135450128, 135480878, 135585059, 135733571,
        135829644, 135873641, 135897044, 135959796, 135963127, 136057351,
        136071691, 136109813, 136118485, 136199448, 136225985, 136265173,
        136304047, 136511648, 136553435, 136687414, 136707734, 136956177,
        137050557, 137069083, 137094337, 137207414, 137304704, 137559424,
        137579558, 137730476, 137792110, 137901134, 137956876, 138208064,
        138293672, 138364384, 138372707, 138440349, 138476054, 138547652,
        138610594, 138655653, 138707984, 138809814, 139081847, 139103026,
        139138456, 139152117, 139274789, 139306912, 139367065, 139368547,
        139699497, 139735827, 139827703, 139899169, 140015620, 140047362,
        140120030, 140149070, 140284181, 140382930, 140435393, 140440661,
        140456146, 140512641, 140601269, 140648319, 140684430, 140765355,
        141006575, 141091655, 141252218, 141450918, 141458906, 141594572,
        141805043, 141994201, 142025122, 142142862, 142162132, 142186427,
        142195060, 142232660, 142276758, 142412698, 142497171, 142724184,
        142746131, 142777489, 142793656, 142797159, 142876994, 142952837,
        142974116, 143025802, 143119159, 143125333, 143138923, 143224246,
        143311535, 143315034, 143375266, 143390689, 143633943, 143793260,
        144243483, 144290099, 144528320, 144530071, 144544602, 144588540,
        144628732, 144668295, 144957750, 144971216, 144975385, 145006503,
        145172305, 145327838, 145363322, 145640708, 145701059, 145716788,
        145956190, 145961485, 146271626, 146348409, 146358927, 146416827,
        146429925, 146542814, 146715489, 146778883, 146883510, 146906914,
        147009100, 147040615, 147382964, 147497595, 147517701, 147977931,
        147986968, 148125056, 148275234, 148310609, 148390672, 148436278,
        148494696, 148597141, 148669246, 148749788, 148801978, 148813832,
        148835046, 148926075, 148930449, 149114189, 149119906, 149284063,
        149432376, 149647349, 149655670, 149671382, 149877327, 149879425,
        149958304, 149973993]

    return out_of_plane_seeds, in_plane_seeds

if __name__ == "__main__":
    main()
