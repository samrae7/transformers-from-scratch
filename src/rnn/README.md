## Milestone 1: RNN where the full generated sequence it passed back in each time

Example of generated results:

```
avg_val_loss=1.9722430300712586
generated:
Wherefore art thou Romeo? saver it A citinghamin us fault the vience so bost nexitert weal my many roburrent thousey?

JULIENGET:
Tespither nuct me of shal he so morcess of your was no mas, whitistany lipelimmon, of ske?

CLIUS:
Why'er'word Moth thou so fear thy paledy.

NECSO:
By mady laredy Juch eit or I biding so Gleateret, thee, 'Tear; in unto the beasur heserved id years, in for hons'eous:
Wemat and so,
That's was the will have my herize o' then before a lordpage, and eit
Marce whe whoutter, be of life, their housely your country reyet?
Now, I
MARD IV:
All powerer aughtry, was one but woukitege call I mimery you not dolenw come tempalt in even the was newtent me to the day, I pray made meay till I me worse: think athat dhose thou'd. A would weal
My weros.

LEONT:
Who baught-sleft thus justreath than full'ment
Hat princemitied, sless print kinglay, I would of Lauls thee; and shadls, and been hy reman
Hat namet's good viyost ous upon
Best or eye of the heach resent it what with
rith's soul answeance; sors:
```

Then instead of passing the generated text back in on each succesive model call, only pass back in the accumulated hidden state, but I made a mistake and got this:

```
avg_val_loss=1.969434335231781
generated:

Thas ushineayowith thy, Ed say d sppe ys thay in'd at mste wofiondinthag, heamfry I tan scum IAbuis t br ht.
O:
Wthingitongred st fat be
Ast moumofothor wand lowo cownarcot wod,
WANd w-bouean GHe s MINlis d teestortoncke n hatobofouth, tous D's wext:


G: pr arthand, m; chixtlamed t fovor hom LOfoue, lar cs st, he f iesulino winthie, Late Gus r bl a my sad
I an cofre stheror th it lolige thaly that

Bypr cest l d; ist, t this sacalll is thathind BASARapay wounges t, m hereyeve d Wam t, desy t ELUSite, ht benes!
ARTICARUSe pour my dr bs me.

Towors grrd athorthat wroul sano courod
Thig
HA: heanesther s
ASholed ke mrsee'Thagas!
F he.
ENil taigr ENGO swe t mas wakelit ilt gor id, ut f way h poun gsou htomomyse:
MO:
Anglit ar meawnd wamauspanedenk sthes storaino and busombo gthathfass withavesak.
Anghellond hesyoas hindind sou anous d hout, R:
S:? in,
Tounee thed d e, fee beldse y hant
He l wanbef thyoustethors Mo-
Sorowanesh oey p usstht huspr an mereras bl:
ELILort hu REND athacke om dou
```

The mistake I'd made was not passing the hidden state passed into forward, and still initing a new hidden state for each `forward` called

I had bumped embed dim from 10 -> 64, and hidden dim from 124 -> 248

I got this

```
HENRY ANIO:
Why: nay, along piteen whine the come. Hapey woed oipons of jury, but you'll cause at
Nor the dear his breath.

BUCKINGHARD IA:
As us, and to keep forward with dost drinkings;
For the questest; for this man epour do wake that yeir, and suit;
I tell heaven,--
Nor whoar to the near him hopy wrongs, I know to' Pamels:
And, in dim life, with of Nurtusave you having traitor: learful see a too
him the morness shepherd!' fair.
Ho hither sevies to seek thy venough in redomaninal more blor father, would in the mield you down knowart, how you, and requile: but is speak do?

SICINIUS:
We had these belove to do
men, whereford's lips, look with lie to telcom to heave their fany wismen to heart
That way wale me way.

ISABELLA:
Now plour, and cleap it
Throught is dectixfost his torch swear very, I have there, need of me; who chargain his put deuch cold eyes a plade
On your cry than trixecunce Edward's this brother?
There
Displease the harring, in this may egence,
Of sheposees are
My nuth
```

to compare like with like I changed those back and got

```
SHoth
Rut in the not hath nibey bitt!

as false toor her his but forth your prince, to may hear her.

SifuTannes
ecriediey,
And pent,
In she cleap,
Docles,
bends as bay the would as of I how I action.

QUEEN MARIOLYE:
Or fould tear's most knest poan
Hadment to then; we for death?
That her, guhe make,
To out would reppittery were drow ank Iurrows
To be gentle-dis from reposh of Lords poise you stion the my grove-have right shall wards lange the for toling spliete; and than my geard; times; but I wall bid to my fice:
To my heaven on them wake't but who hise all.
Match of the clraned, Kight, and is the not standelign
And us father,
A well dient by tell your for ere death bould me. goade; whenc' and long'd
another my do call, by consilst I were prinial all
Both:
I'll your wuall, would hile for, I
Losk beficona-fomes,
For gree:
Unkeword!

KING RICHARD:
A; preeebty her he made your cloud DO:
lfo thinds but, pixe, and move, good offection;
When vanty:
I
Jurth and
By rene
And some fuld on, ar
```


Then:

```
embed_dim = 32
hidden_dim = 128
sequence_length = 32
batch_size = 32
learning_rate = 1e-3
```

NB: sequence length bumped 8 -> 32 so model can learn from structure across longer text lengths.

```
avg_val_loss=1.742593207359314
generated:

Of thyse
Where am I know they rather, hand!
God the stummbricoment and stock desprid yours, for my blutt othrill? It a cuch vento that his one as I am young enemiot teabs love you, here is consul much tors life barrower; onders you spuckening,
Thrums; but site
Leasure.

RICILIZEL:
'Thark'd:
Ho:
In were dear him man thy lord is; and lobif is to take meth,
you take descarst to-mout.

HERMIONE:
I have a fined that devil,
Our ear,
If brother in this leay of enemyou and bay mace
I have
good
is I know the grains, faye,
I thank you, her death:
A lady to the city.

COPION:
Still; fell, On see of Gudyentition hath true-law, book, chast: how Capcubable.
Do, Julious boddliness you as bread,
Which resed.

GLOUCESTER:
There's teder hath I is presaint of Lord.

JOWI hearthy dead,
But luckled ansid: childirgh?

KADWARD II:
Hold of truedy in his great say be my dupt
Incembercely.

Dutwition! Undown: ou storsors when it, to-morrone mournger:
His heen be her, and reasiaring. I'll be bread wonting, be at
```

