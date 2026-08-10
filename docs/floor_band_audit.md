# Face-validity audit of the 20–49 band (milestone M23.3)

`python scripts/audit_floor_band.py --anchors 8 --seed 42` — 8 anchors drawn uniformly from the **20–49 reader** band, the band the shipped anchor floor of 50 closes and a floor of 20 would open. 8 anchors x 3 configurations x 10 slots = **240 slots** to read.

Configurations: **A** = ALS (what ships), **B** = item-item shrunk cosine, **C** = RRF over item-item + TF-IDF. The candidate floor stays at 20 in all three (L34); only the engine changes.

**Count a slot bad when it is** (L79's criterion, inherited unchanged so the two counts are comparable): the anchor itself under another edition, title or translation; a companion or *about* book rather than a comparable read; or a duplicate of another slot in the same list.

`co` is the number of the anchor's readers who also read that book — the number the app prints. `score` is each engine's own quantity and is **not comparable across columns**: A is an ALS factor cosine, B a shrunk cosine, C a fused rank sum.

## Being a Green Mother (Incarnations of Immortality (Paperback)) — Piers Anthony

`being a green mother|anthony` · **39 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | Wielding a Red Sword (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 16 | Wielding a Red Sword (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 16 | Wielding a Red Sword (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 16 |
| 2 | Bearing an Hourglass (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 19 | With a Tangled Skein (Incarnations of Immortality (Paperback)) — *PIERS ANTHONY* | 18 | With a Tangled Skein (Incarnations of Immortality (Paperback)) — *PIERS ANTHONY* | 18 |
| 3 | And Eternity (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 12 | Bearing an Hourglass (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 19 | And Eternity (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 12 |
| 4 | For Love of Evil : Book Six of Incarnations of Immortality (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 14 | For Love of Evil : Book Six of Incarnations of Immortality (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 14 | For Love of Evil : Book Six of Incarnations of Immortality (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 14 |
| 5 | With a Tangled Skein (Incarnations of Immortality (Paperback)) — *PIERS ANTHONY* | 18 | And Eternity (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 12 | Bearing an Hourglass (Incarnations of Immortality (Paperback)) — *Piers Anthony* | 19 |
| 6 | On a Pale Horse (Incarnations of Immortality, Bk. 1) — *Piers Anthony* | 17 | On a Pale Horse (Incarnations of Immortality, Bk. 1) — *Piers Anthony* | 17 | On a Pale Horse (Incarnations of Immortality, Bk. 1) — *Piers Anthony* | 17 |
| 7 | Split Infinity — *Piers Anthony* | 9 | Split Infinity — *Piers Anthony* | 9 | Virtual Mode (Mode (Paperback)) — *Piers Anthony* | 6 |
| 8 | Xanth 14: Question Quest — *Piers Anthony* | 6 | Juxtaposition (Apprentice Adept (Paperback)) — *Piers Anthony* | 7 | Juxtaposition (Apprentice Adept (Paperback)) — *Piers Anthony* | 7 |
| 9 | Dragon on a Pedestal — *Piers Anthony* | 6 | OUT OF THE SILENT PLANET — *C.S. Lewis* | 7 | Split Infinity — *Piers Anthony* | 9 |
| 10 | Blue Adept — *Piers Anthony* | 7 | Blue Adept — *Piers Anthony* | 7 | Blue Adept — *Piers Anthony* | 7 |

## Songlines — Bruce Chatwin

`songlines|chatwin` · **36 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | He Died with a Felafel in His Hand — *John Birmingham* | 1 | The Spy Who Came In from the Cold — *John le Carre* | 3 | English Passengers — *Matthew Kneale* | 3 |
| 2 | Vernon God Little: A 21st Century Comedy in the Presence of Death — *D. B. C. Pierre* | 3 | Man Who Mistook His Wife for a Hat — *Oliver Sacks* | 3 | The Woman in White (The Penguin English Library) — *Wilkie Collins* | 3 |
| 3 | Miss Smilla's Feeling for Snow — *Peter Hoeg* | 1 | English Passengers — *Matthew Kneale* | 3 | Lady Chatterley's Lover — *D.H. Lawrence* | 2 |
| 4 | Eats, Shoots and Leaves: The Zero Tolerance Approach to Punctuation — *Lynne Truss* | 2 | The New York Trilogy: City of Glass, Ghosts, the Locked Room (Contemporary American Fiction Series) — *Paul Auster* | 3 | Bleak House (English Library) — *Charles Dickens* | 2 |
| 5 | Man Who Mistook His Wife for a Hat — *Oliver Sacks* | 3 | Oscar and Lucinda — *Peter Carey* | 3 | The Spy Who Came In from the Cold — *John le Carre* | 3 |
| 6 | Larrys Party — *Carol Shields* | 1 | Our Father — *Marilyn French* | 2 | The Songlines — *Bruce Chatwin* | 1 |
| 7 | The Bride Stripped Bare — *Anonymous* | 0 | Knowledge of Angels — *Jill Paton Walsh* | 2 | Man Who Mistook His Wife for a Hat — *Oliver Sacks* | 3 |
| 8 | Remains of the Day — *Kazuo Ishiguro* | 1 | Vernon God Little: A 21st Century Comedy in the Presence of Death — *D. B. C. Pierre* | 3 | The Piper's Sons — *Bruce Chandler Fergusson* | 0 |
| 9 | Horse Whisperer — *Nicholas Evans* | 1 | The Tin Princess — *Philip Pullman* | 2 | My Teacher Flunked the Planet — *Bruce Coville* | 0 |
| 10 | The Songlines — *Bruce Chatwin* | 1 | Swim With the Sharks: Without Being Eaten Alive : Outsell, Outmanage, Outmotivate, and Outnegotiate Your Competition — *Harvey Mackay* | 2 | Holy Fire: A Novel (Bantam Spectra Book) — *Bruce Sterling* | 0 |

## Love — Leo Buscaglia

`love|buscaglia` · **34 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | SEVEN HABITS OF HIGHLY EFFECTIVE PEOPLE : Powerful Lessons in Personal Change — *Stephen R. Covey* | 2 | If Life Is a Bowl of Cherries What Am I — *Erma Bombeck* | 6 | My Sergei : A Love Story — *E. M. Swift* | 4 |
| 2 | Death in a Tenured Position (Kate Fansler Novels (Paperback)) — *Amanda Cross* | 4 | On Death & Dying — *Kubler Elisabeth Ross* | 5 | If Life Is a Bowl of Cherries What Am I — *Erma Bombeck* | 6 |
| 3 | Women Who Run with the Wolves — *CLARISSA PINKOLA PHD ESTES* | 5 | Swim With the Sharks: Without Being Eaten Alive : Outsell, Outmanage, Outmotivate, and Outnegotiate Your Competition — *Harvey Mackay* | 5 | For Love — *Sue Miller* | 3 |
| 4 | Lila: An Inquiry Into Morals — *Robert M. Pirsig* | 1 | Oliver's Story — *Erich Segal* | 5 | On Death & Dying — *Kubler Elisabeth Ross* | 5 |
| 5 | You Can Heal Your Life/101 — *Louise L. Hay* | 2 | It's Always Something — *Gilda Radner* | 8 | PS, I Love You — *Cecelia Ahern* | 1 |
| 6 | How to Talk So Kids Will Listen and Listen So Kids Will Talk — *Adele Faber* | 3 | The Fourth Protocol — *Frederick Forsyth* | 5 | Swim With the Sharks: Without Being Eaten Alive : Outsell, Outmanage, Outmotivate, and Outnegotiate Your Competition — *Harvey Mackay* | 5 |
| 7 | Your Erroneous Zones — *Wayne W. Dyer* | 2 | The Cat Who Played Post Office (Cat Who... (Paperback)) — *Lilian Jackson Braun* | 6 | Love and War — *John Jakes* | 1 |
| 8 | The Physics of Star Trek — *Lawrence M. Krauss* | 2 | Final Flight — *Stephen Coonts* | 5 | Oliver's Story — *Erich Segal* | 5 |
| 9 | When I Am an Old Woman I Shall Wear Purple — *Sandra Martz* | 4 | The Doctor's Book of Home Remedies : Thousands of Tips and Techniques Anyone Can Use to Heal Everyday Health Problems — *Editors Prevention Health Books* | 4 | Love — *TONI MORRISON* | 0 |
| 10 | The New New Thing: A Silicon Valley Story — *Michael Lewis* | 1 | A Man Rides Through (Man Rides Through) — *Stephen Donaldson* | 5 | It's Always Something — *Gilda Radner* | 8 |

## The Funny Thing Is... — Ellen DeGeneres

`the funny thing is...|degeneres` · **33 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | Dry: A Memoir — *Augusten Burroughs* | 4 | The Second Sex — *Simone De Beauvoir* | 4 | My Point - And I Do Have One — *Ellen Degeneres* | 3 |
| 2 | Welcome to the Great Mysterious (Ballantine Reader's Circle) — *LORNA LANDVIK* | 2 | Woman: An Intimate Geography — *Natalie Angier* | 4 | The Fifth Sacred Thing — *Starhawk* | 2 |
| 3 | I'm the One That I Want — *Margaret Cho* | 4 | Letters to a Young Poet — *Rainer Maria Rilke* | 4 | The Dim Sum of All Things — *Kim Wong Keltner* | 2 |
| 4 | Going Down — *Jennifer Belle* | 3 | I'm the One That I Want — *Margaret Cho* | 4 | The Second Sex — *Simone De Beauvoir* | 4 |
| 5 | The Romance Reader — *Pearl Abraham* | 2 | Cat (Wildflowers) — *V.C. Andrews* | 4 | Woman: An Intimate Geography — *Natalie Angier* | 4 |
| 6 | A Slipping-Down Life — *ANNE TYLER* | 2 | Fingersmith — *Sarah Waters* | 6 | My Point...And I Do Have One — *ELLEN DEGENERES* | 1 |
| 7 | Property of — *Alice Hoffman* | 2 | Star (Wildflowers) — *V.C. Andrews* | 4 | Letters to a Young Poet — *Rainer Maria Rilke* | 4 |
| 8 | Sellevision : A Novel — *Augusten Burroughs* | 2 | The Love Letter — *Cathleen Schine* | 5 | Westing Game — *Ellen Raskin* | 0 |
| 9 | The Second Sex — *Simone De Beauvoir* | 4 | Dress Codes: Of Three Girlhoods--My Mother's, My Father's, and Mine — *Noelle Howey* | 3 | I'm the One That I Want — *Margaret Cho* | 4 |
| 10 | High Maintenance — *Jennifer Belle* | 5 | Nights in Rodanthe — *Nicholas Sparks* | 7 | Funny in Farsi : A Memoir of Growing Up Iranian in America — *FIROOZEH DUMAS* | 0 |

## Beauty: A Retelling of the Story of Beauty and the Beast — Robin McKinley

`beauty: a retelling of the story of beauty and the beast|mckinley` · **32 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | The Outlaws of Sherwood — *Robin McKinley* | 5 | The Outlaws of Sherwood — *Robin McKinley* | 5 | The Outlaws of Sherwood — *Robin McKinley* | 5 |
| 2 | The Vor Game — *Lois McMaster Bujold* | 1 | Criminal Seduction — *Darian North* | 4 | Deerskin — *Robin McKinley* | 3 |
| 3 | A Game of Thrones (A Song of Ice and Fire, Book 1) — *George R.R. Martin* | 6 | Rescuing Rose (Red Dress Ink) — *Isabel Wolff* | 4 | The Blue Sword — *Robin McKinley* | 3 |
| 4 | Prospero's Children — *Jan Siegel* | 4 | Sirena — *Donna Jo Napoli* | 4 | The Hero and the Crown — *Robin McKinley* | 3 |
| 5 | The Virgin's Knot — *Holly Payne* | 2 | Bridge of Birds — *Barry Hughart* | 4 | Criminal Seduction — *Darian North* | 4 |
| 6 | The Hero and the Crown — *Robin McKinley* | 3 | The Magic of Recluce (Recluce series, Book 1) — *L. E. Modesitt Jr.* | 4 | Rescuing Rose (Red Dress Ink) — *Isabel Wolff* | 4 |
| 7 | The Man in the Iron Mask (Signet Regency Romance) — *Alexandre Dumas* | 3 | The Dispossessed: An Ambiguous Utopia — *Ursula K. Le Guin* | 4 | Rose Daughter — *Robin McKinley* | 0 |
| 8 | Bridge of Birds — *Barry Hughart* | 4 | Zia — *Scott O'Dell* | 4 | Sirena — *Donna Jo Napoli* | 4 |
| 9 | The Church of Dead Girls: A Novel — *Stephen Dobyns* | 2 | A Game of Thrones (A Song of Ice and Fire, Book 1) — *George R.R. Martin* | 6 | Spindle's End — *Robin McKinley* | 2 |
| 10 | Five Fortunes — *Beth Gutcheon* | 1 | From Beginning to End: The Rituals of Our Lives — *Robert Fulghum* | 4 | Bridge of Birds — *Barry Hughart* | 4 |

## Thank You for Smoking — Christopher Buckley

`thank you for smoking|buckley` · **26 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | Little Green Men : A Novel — *Christopher Buckley* | 2 | Take the Cannoli : Stories From the New World — *Sarah Vowell* | 6 | Take the Cannoli : Stories From the New World — *Sarah Vowell* | 6 |
| 2 | Who Will Run the Frog Hospital? — *Lorrie Moore* | 2 | A Yellow Raft in Blue Water — *Michael Dorris* | 7 | Little Green Men : A Novel — *Christopher Buckley* | 2 |
| 3 | Take the Cannoli : Stories From the New World — *Sarah Vowell* | 6 | The Chancellor Manuscript — *Robert Ludlum* | 5 | A Yellow Raft in Blue Water — *Michael Dorris* | 7 |
| 4 | Comfort Me with Apples: More Adventures at the Table — *RUTH REICHL* | 2 | Trouble in Paradise (Jesse Stone Novels (Paperback)) — *Robert B. Parker* | 5 | The Cold One — *Christopher Pike* | 0 |
| 5 | The All-True Travels and Adventures of Lidie Newton: A Novel (Ballantine Reader's Circle) — *Jane Smiley* | 3 | Cat's Meow: A Novel — *Melissa de la Cruz* | 4 | The Chancellor Manuscript — *Robert Ludlum* | 5 |
| 6 | You Are Not a Stranger Here (Today Show Book Club #2) — *ADAM HASLETT* | 3 | Dating Big Bird — *Laura Zigman* | 6 | Sati — *Christopher Pike* | 0 |
| 7 | Beyond Suspicion — *James Grippando* | 3 | Life on the Mississippi (Bantam Classics) — *Mark Twain* | 4 | Trouble in Paradise (Jesse Stone Novels (Paperback)) — *Robert B. Parker* | 5 |
| 8 | The Partly Cloudy Patriot — *Sarah Vowell* | 4 | The Partly Cloudy Patriot — *Sarah Vowell* | 4 | The Snow Garden — *Christopher Rice* | 1 |
| 9 | In a Sunburned Country — *Bill Bryson* | 5 | TENDER IS THE NIGHT — *F. Scott Fitzgerald* | 5 | Cat's Meow: A Novel — *Melissa de la Cruz* | 4 |
| 10 | Barrel Fever : Stories and Essays (Barrel Fever) — *David Sedaris* | 5 | All Creatures Great and Small — *James Herriot* | 6 | Still Me — *Christopher Reeve* | 1 |

## Main Street — Sinclair Lewis

`main street|lewis` · **21 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | Angels & Insects : Two Novellas — *A.S. BYATT* | 1 | Babbitt (Signet Classics (Paperback)) — *Sinclair Lewis* | 4 | Babbitt (Signet Classics (Paperback)) — *Sinclair Lewis* | 4 |
| 2 | Summer of My German Soldier (Law at Work) — *Bette Greene* | 2 | All the King's Men (Harvest Book) — *Robert Penn Warren* | 4 | The Jungle (Bantam Classics) — *Upton Sinclair* | 5 |
| 3 | All the King's Men (Harvest Book) — *Robert Penn Warren* | 4 | Catherine, Called Birdy (Trophy Newbery) — *Karen Cushman* | 5 | Silver on the Tree — *Susan Cooper* | 3 |
| 4 | Who Will Run the Frog Hospital? — *Lorrie Moore* | 1 | Wizard of Oz (Wordsworth Collection) — *L. F. Baum* | 4 | All the King's Men (Harvest Book) — *Robert Penn Warren* | 4 |
| 5 | Bailey's Cafe (Vintage Contemporaries) — *Gloria Naylor* | 2 | Slaughterhouse Five or the Children's Crusade: A Duty Dance With Death — *Kurt Vonnegut* | 7 | Coffee Will Make You Black — *April Sinclair* | 1 |
| 6 | Orlando: A Biography — *Virginia Woolf* | 3 | Scary Stories 3 : More Tales to Chill Your Bones (Scary Stories) — *Alvin Schwartz* | 3 | Catherine, Called Birdy (Trophy Newbery) — *Karen Cushman* | 5 |
| 7 | Mama Day (Vintage Contemporaries) — *Gloria Naylor* | 2 | Before and After — *Rosellen Brown* | 4 | Wizard of Oz (Wordsworth Collection) — *L. F. Baum* | 4 |
| 8 | Gravity's Rainbow (Penguin Twentieth-Century Classics) — *Thomas Pynchon* | 1 | First Things First: To Live, to Love, to Learn, to Leave a Legacy — *Stephen R. Covey* | 3 | The Silver Chair — *C. S. Lewis* | 1 |
| 9 | A People's History of the United States : 1492-Present (Perennial Classics) — *Howard Zinn* | 3 | The Dancing Floor — *Barbara Michaels* | 3 | Slaughterhouse Five or the Children's Crusade: A Duty Dance With Death — *Kurt Vonnegut* | 7 |
| 10 | Growing Up — *Russell Baker* | 3 | Mrs Dalloway — *Virginia Woolf* | 6 | The Four Loves — *C. S. Lewis* | 0 |

## The Samurai's Wife (A Sano Ichiro Mystery) — Laura Joh Rowland

`the samurai's wife|rowland` · **20 readers**

| # | A · ALS | co | B · item-item | co | C · RRF | co |
|---|---|---:|---|---:|---|---:|
| 1 | The Skull Beneath the Skin — *P. D. James* | 3 | Black Lotus (A Sano Ichiro Mystery) — *Laura Joh Rowland* | 4 | Black Lotus (A Sano Ichiro Mystery) — *Laura Joh Rowland* | 4 |
| 2 | The Physics of Star Trek — *Lawrence M. Krauss* | 2 | Gibbon's Decline and Fall — *Sheri S. Tepper* | 4 | The Salaryman's Wife (Children of Violence Series) — *Sujata Massey* | 3 |
| 3 | A Grave Talent — *Laurie R. King* | 2 | A Caress of Twilight (Meredith Gentry Novels (Paperback)) — *LAURELL K. HAMILTON* | 5 | Gibbon's Decline and Fall — *Sheri S. Tepper* | 4 |
| 4 | The New New Thing: A Silicon Valley Story — *Michael Lewis* | 1 | Tapestry — *Belva Plain* | 4 | Shinju — *Laura Joh Rowland* | 2 |
| 5 | Quincunx — *Charles Palliser* | 1 | Winterdance: The Fine Madness of Running the Iditarod — *Gary Paulsen* | 3 | A Caress of Twilight (Meredith Gentry Novels (Paperback)) — *LAURELL K. HAMILTON* | 5 |
| 6 | The Evening News — *Arthur Hailey* | 2 | The Hippopotamus — *Stephen Fry* | 3 | The Samurai's Garden : A Novel — *Gail Tsukiyama* | 1 |
| 7 | Infinite Jest : A Novel — *David Foster Wallace* | 2 | Island of the Sequined Love Nun — *Christopher Moore* | 4 | Tapestry — *Belva Plain* | 4 |
| 8 | Winterdance: The Fine Madness of Running the Iditarod — *Gary Paulsen* | 3 | Interesting Times (Discworld Novels (Paperback)) — *Terry Pratchett* | 3 | Shades of Earl Grey (A Tea Shop Mystery) — *Laura Childs* | 0 |
| 9 | London Match — *Len Deighton* | 0 | Enter Whining — *Fran Drescher* | 3 | Winterdance: The Fine Madness of Running the Iditarod — *Gary Paulsen* | 3 |
| 10 | The Jungle Book (Wordsworth Collection) — *Rudyard Kipling* | 2 | Seduced By Moonlight (Hamilton, Laurell K) — *LAURELL K. HAMILTON* | 3 | Keepsake Crimes (First Scrapbooking Mystery) — *Laura Childs* | 0 |

## The count

Counted by hand by hand on 2026-08-10, reading all 240 slots above, the same
way L42's 30 clusters and L79's 30 gallery slots were counted.

| configuration | bad slots (L79's criterion) | text-match artefacts | of | counted by | date |
|---|---:|---:|---:|---|---|
| A · ALS | **1** | 0 | 80 | hand count | 2026-08-10 |
| B · item-item | **0** | 0 | 80 | hand count | 2026-08-10 |
| C · RRF | **2** | **17** | 80 | hand count | 2026-08-10 |

**The three bad slots, named, so the count can be checked rather than believed.**

- **A, *Songlines* slot 10** — *The Songlines* by Bruce Chatwin: the anchor itself under a
  leading article the work key does not strip. Criterion 1.
- **C, *Songlines* slot 6** — the identical defect, three ranks higher.
- **C, *The Funny Thing Is...* slots 1 and 6** — *My Point - And I Do Have One* and *My
  Point...And I Do Have One*, the same Ellen DeGeneres book under two punctuations. Criterion
  3, and it is L47's edition key failing on punctuation rather than on translation.

**The second column is new in M23 and is defined here rather than borrowed, because L79's
criterion cannot see this failure mode and it is the dominant one in this band.** A *text-match
artefact* is a slot that shares a title word or **a name token from the anchor author's name —
first or last** with the anchor, carries **at most one co-reader**, and is not a comparable
read. It is reported separately and never folded into the first column, so the L79 comparison
stays exact. *(This definition first read "the author's **first** name". That is the dominant
case but not all of it: two of the seventeen — C. S. **Lewis** against Sinclair **Lewis** —
match on the surname. Corrected 2026-08-10 to describe what was actually counted. The count of
17 is unchanged and no uncounted slot newly qualifies.)*

All 17 belong to C, and they are the same mechanism seventeen times — TF-IDF matching on a
name token that carries no meaning:

| anchor | the artefacts C returns |
|---|---|
| *Songlines* (Bruce Chatwin) | *The Piper's Sons* (Bruce **Chandler Fergusson**), *My Teacher Flunked the Planet* (Bruce **Coville**), *Holy Fire* (Bruce **Sterling**) — all 0 co-readers |
| *Love* (Leo Buscaglia) | *PS, I Love You*, *Love and War*, *Love* (Toni Morrison, 0 co-readers) |
| *The Funny Thing Is...* (Ellen DeGeneres) | *Westing Game* (**Ellen** Raskin, 0), *Funny in Farsi* (0) |
| *Thank You for Smoking* (Christopher Buckley) | *The Cold One* and *Sati* (Christopher **Pike**, 0 each), *The Snow Garden* (Christopher **Rice**), *Still Me* (Christopher **Reeve**) |
| *Main Street* (Sinclair Lewis) | *Coffee Will Make You Black* (April **Sinclair**), *The Silver Chair* and *The Four Loves* (C. S. **Lewis**) |
| *The Samurai's Wife* (Laura Joh Rowland) | *Shades of Earl Grey* and *Keepsake Crimes* (**Laura** Childs, 0 each) |

**Read against the accuracy column, this is the whole M23 finding in one table.** C is the
configuration L85 ranks first on the item query, and on the band a lower floor would open it
it puts books with **zero shared readers** in front of a reader **12 times in 80 slots, against
A's 2 and B's 0** — because it matched *Christopher* Buckley to *Christopher* Pike. B returns no
bad slot, no artefact and no zero-evidence slot at all, and where the anchor has a real
neighbourhood it finds it (*Babbitt* for *Main Street*, *Black Lotus* for *The Samurai's Wife*,
the whole *Incarnations of Immortality* sequence for *Being a Green Mother*).

*(This paragraph first said C was **the only one** of the three to do it. That is false against
the table directly above it: A returns two zero-co-reader slots — *The Bride Stripped Bare* for
*Songlines*, *London Match* for *The Samurai's Wife*. Corrected 2026-08-10. The finding survives
as a rate rather than a category — 15.0% against 2.5% against 0.0% — and the uniqueness that
does hold belongs to **B**, which never does it.)*

**In fairness to C, and recorded because the count is not the whole picture:** on *Beauty: A
Retelling of the Story of Beauty and the Beast* C returns five Robin McKinley novels including
*Rose Daughter*, her other Beauty-and-the-Beast retelling, which is the single best slot any
configuration produces anywhere in this table — and it has 0 co-readers, so no collaborative
engine could ever have found it. C's failure mode and C's best moment are the same mechanism.

